import unittest

from autodex.tasks import (
    LiftTask,
    TaskContext,
    TaskOutcome,
    attach_task_outcome,
)


class LiftTaskTest(unittest.TestCase):
    def setUp(self):
        self.context = TaskContext(
            object_name="precision_key_1p5",
            arm="franka",
            hand="inspire",
            trial_dir="/tmp/trial",
            scene_info=("table", "0", "7"),
        )

    def test_lift_task_preserves_existing_success_semantics(self):
        outcome = LiftTask().evaluate(
            context=self.context,
            grasp_success=True,
            grasp_evidence={"source": "charuco"},
        )
        record = attach_task_outcome(
            {"dir_idx": "trial"},
            grasp_success=True,
            task_outcome=outcome,
        )

        self.assertTrue(record["grasp_success"])
        self.assertTrue(record["task_success"])
        self.assertTrue(record["success"])
        self.assertEqual(record["task"]["name"], "grasp_lift")
        self.assertEqual(record["task"]["status"], "success")

    def test_task_failure_does_not_relabel_successful_grasp(self):
        outcome = TaskOutcome(
            task_name="precision_insertion",
            success=False,
            reason="socket_contact_without_insertion",
        )
        record = attach_task_outcome(
            {}, grasp_success=True, task_outcome=outcome)

        self.assertTrue(record["grasp_success"])
        self.assertFalse(record["task_success"])
        self.assertFalse(record["success"])

    def test_unjudgeable_is_preserved(self):
        outcome = LiftTask().evaluate(
            context=self.context, grasp_success=None)
        record = attach_task_outcome(
            {}, grasp_success=None, task_outcome=outcome)

        self.assertIsNone(record["grasp_success"])
        self.assertIsNone(record["task_success"])
        self.assertEqual(record["task"]["status"], "unjudgeable")


if __name__ == "__main__":
    unittest.main()
