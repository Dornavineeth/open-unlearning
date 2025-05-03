import logging
from lm_eval.models.hf_vlms import HFLM
from lm_eval.tasks import TaskManager
from lm_eval import simple_evaluate

from evals.base import Evaluator


logger = logging.getLogger("evaluator")


class LMEvalEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        self.name = "LMEval"
        self.eval_cfg = eval_cfg
        self.tasks = list(self.eval_cfg.tasks)
        self.task_manager = TaskManager()
        self.simple_evaluate_args = kwargs.get("simple_evaluate_args", {})

    def prepare_model(self, model, **kwargs):
        """Prepare model for evaluation"""
        model.eval()
        return HFLM(model)

    def summarize(self, results):
        """Summarize the metrics results"""
        metric_summary = {}
        if not isinstance(results, dict):
            # Unexpected format; return empty to be safe
            return metric_summary
        # Unwrap if results are nested under a 'results' key
        if "results" in results and isinstance(results["results"], dict):
            results = results["results"]

        for task, metrics in results.items():
            # Each task entry should be a dict of metrics
            if not isinstance(metrics, dict):
                continue
            for metric_name, value in metrics.items():
                try:
                    numeric_val = float(value)
                except (TypeError, ValueError):
                    continue
                # Add to flat dict with the prefixed key
                flat_key = f"{task}/{metric_name}"
                metric_summary[flat_key] = numeric_val
        return metric_summary

    def get_task_name(self, task):
        if isinstance(task, str):
            return task
        elif isinstance(task, dict):
            if "task" in task:
                return task.get("task")
        raise ValueError(f"Invalid task format: {task}")

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        # set flag to overwrite metrics
        overwrite = self.eval_cfg.overwrite if overwrite is None else overwrite

        # Prepare model for evaluation
        kwargs = {"tokenizer": kwargs.get("tokenizer", None)}
        model = self.prepare_model(model, **kwargs)

        # Set output_dir and file to store results
        output_dir = output_dir if output_dir else self.eval_cfg.output_dir
        logs_file_path = self.get_logs_file_path(output_dir)
        summary_file_path = self.get_logs_file_path(output_dir, suffix="SUMMARY")

        # Load existing results from file if any.
        logs = self.load_logs_from_file(logs_file_path) if not overwrite else {}
        summary = self.load_logs_from_file(summary_file_path) if not overwrite else {}

        logger.info(f"***** Running {self.name} evaluation suite *****")
        logger.info(f"Fine-grained evaluations will be saved to: {logs_file_path}")
        logger.info(
            f"Aggregated evaluations will be summarised in: {summary_file_path}"
        )

        for task in self.tasks:
            task_name = self.get_task_name(task)
            if not overwrite and task_name in logs and logs[task_name]:
                logger.info(f"Skipping {task_name}, already evaluated.")
                continue
            _ = logs.pop(task_name, None)  # overwriting existing evals if present
            results = simple_evaluate(
                model=model,
                tasks=[task],
                task_manager=self.task_manager,
                **self.simple_evaluate_args,
            )
            logs.update({task_name: results["samples"]})
            summary.update(self.summarize(results))
            self.save_logs(logs, logs_file_path)
            self.save_logs(summary, summary_file_path)
        return summary
