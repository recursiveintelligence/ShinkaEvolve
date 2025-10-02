from typing import List, Optional, Tuple
import numpy as np
from shinka.database import Program
from shinka.prompts import (
    construct_eval_history_msg,
    perf_str,
    format_text_feedback_section,
    BASE_SYSTEM_MSG,
    DIFF_SYS_FORMAT,
    DIFF_ITER_MSG,
    FULL_ITER_MSG,
    FULL_SYS_FORMATS,
    CROSS_SYS_FORMAT,
    CROSS_ITER_MSG,
    get_cross_component,
)
from shinka.prompts.prompts_init import INIT_SYSTEM_MSG, INIT_USER_MSG
import logging

logger = logging.getLogger(__name__)


class PromptSampler:
    def __init__(
        self,
        task_sys_msg: Optional[str] = None,
        language: str = "python",
        patch_types: Optional[List[str]] = None,
        patch_type_probs: Optional[List[float]] = None,
        use_text_feedback: bool = False,
    ):
        if patch_types is None:
            patch_types = ["diff"]
        if patch_type_probs is None:
            patch_type_probs = [1.0]

        self.task_sys_msg = task_sys_msg
        self.language = language
        self.patch_types = patch_types
        self.patch_type_probs = patch_type_probs
        # Check if probabilities sum to 1.0 w. tolerance for errors
        prob_sum = np.sum(patch_type_probs)
        if not np.isclose(prob_sum, 1.0, atol=1e-6):
            raise ValueError(
                f"Coding type probabilities must sum to 1.0, got {prob_sum:.6f}"
            )
        # Whether to use text feedback in the prompt
        self.use_text_feedback = use_text_feedback

    def initial_program_prompt(self) -> Tuple[str, str]:
        """Generate the prompt for the initial program."""
        if self.task_sys_msg is None:
            sys_msg = INIT_SYSTEM_MSG
            task_description = "The user has not provided a task description."
        else:
            sys_msg = self.task_sys_msg
            task_description = self.task_sys_msg

        user_msg = INIT_USER_MSG.format(
            language=self.language,
            task_description=task_description,
        )
        return sys_msg, user_msg

    def sample(
        self,
        parent: Program,
        archive_inspirations: List[Program],
        top_k_inspirations: List[Program],
        meta_recommendations: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        if self.task_sys_msg is None:
            sys_msg = BASE_SYSTEM_MSG
        else:
            sys_msg = self.task_sys_msg

        # Sample coding type
        # Filter out crossover if no inspirations
        if len(archive_inspirations) == 0 and len(top_k_inspirations) == 0:
            valid_types = [t for t in self.patch_types if t != "cross"]
            valid_probs = [
                p
                for t, p in zip(self.patch_types, self.patch_type_probs)
                if t != "cross"
            ]
            # Renormalize probabilities
            valid_probs = [p / sum(valid_probs) for p in valid_probs]
            patch_type = np.random.choice(valid_types, p=valid_probs)
        else:
            patch_type = np.random.choice(
                self.patch_types,
                p=self.patch_type_probs,
            )

        if patch_type == "diff":
            sys_msg += DIFF_SYS_FORMAT
        elif patch_type == "full":
            # Randomly sample from different full rewrite variants
            full_variant_idx = np.random.randint(0, len(FULL_SYS_FORMATS))
            selected_format = FULL_SYS_FORMATS[full_variant_idx]
            sys_msg += selected_format
        elif patch_type == "cross":
            sys_msg += CROSS_SYS_FORMAT

        if len(archive_inspirations) > 0:
            eval_history_msg = construct_eval_history_msg(
                archive_inspirations,
                language=self.language,
                include_text_feedback=self.use_text_feedback,
            )
        else:
            eval_history_msg = ""

        # Add top-k inspirations
        # TODO(RobertTLange): Check if order needs inversion
        if len(top_k_inspirations) > 0:
            eval_history_msg += construct_eval_history_msg(
                top_k_inspirations,
                language=self.language,
                include_text_feedback=self.use_text_feedback,
            )

        # Format text feedback section for current program
        text_feedback_section = ""
        if self.use_text_feedback:
            text_feedback_section = "\n" + format_text_feedback_section(
                parent.text_feedback
            )

        if patch_type == "diff":
            iter_msg = DIFF_ITER_MSG.format(
                language=self.language,
                code_content=parent.code,
                performance_metrics=perf_str(
                    parent.combined_score, parent.public_metrics
                ),
                text_feedback_section=text_feedback_section,
            )
        elif patch_type == "full":
            iter_msg = FULL_ITER_MSG.format(
                language=self.language,
                code_content=parent.code,
                performance_metrics=perf_str(
                    parent.combined_score, parent.public_metrics
                ),
                text_feedback_section=text_feedback_section,
            )
        elif patch_type == "cross":
            iter_msg = CROSS_ITER_MSG.format(
                language=self.language,
                code_content=parent.code,
                performance_metrics=perf_str(
                    parent.combined_score, parent.public_metrics
                ),
                text_feedback_section=text_feedback_section,
            )
            iter_msg += "\n\n" + get_cross_component(
                archive_inspirations,
                top_k_inspirations,
                language=self.language,
            )
        elif patch_type == "paper":
            raise NotImplementedError("Paper edit not implemented.")
        else:
            raise ValueError(f"Invalid patch type: {patch_type}")

        # Add meta-recommendations if provided
        sum_rec_msg = ""
        if meta_recommendations not in [None, "none"] and patch_type != "cross":
            sum_rec_msg += "\n\n# Potential Recommendations"
            sum_rec_msg += (
                "\nThe following are potential recommendations for the "
                "next program generations:\n\n"
            )
            sum_rec_msg += f"\n{meta_recommendations}"

        return (
            sys_msg + sum_rec_msg,
            eval_history_msg + "\n" + iter_msg,
            patch_type,
        )
