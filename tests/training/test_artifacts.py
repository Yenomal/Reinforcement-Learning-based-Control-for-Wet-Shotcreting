import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_robot.training.artifacts import save_training_curves, write_metrics_csv


def test_write_metrics_csv_preserves_parallel_training_columns(tmp_path: Path) -> None:
    history = [
        {
            "progress": 20,
            "batch_reward_mean": 0.1,
            "episodes_in_window": 32,
            "success_episodes": 16,
            "success_rate": 0.5,
            "det_success_rate": 0.25,
            "det_mean_min_goal_distance": 0.01,
            "approx_kl": 0.02,
            "explained_variance": 0.99,
            "ppo_success_ema": 0.5,
            "ppo_std_streak": 2.0,
            "ppo_next_log_std": -3.0,
            "ppo_std_cooldown_remaining": 0.0,
            "ppo_std_phase": "trigger",
            "ppo_std_mean": 0.03,
            "ppo_log_std_mean": -3.5,
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "lr": 3e-4,
            "env_steps_per_sec": 1234.0,
        }
    ]

    write_metrics_csv(tmp_path, history)
    content = (tmp_path / "metrics.csv").read_text(encoding="utf-8")

    assert "det_success_rate" in content
    assert "det_mean_min_goal_distance" in content
    assert "approx_kl" in content
    assert "explained_variance" in content
    assert "ppo_success_ema" in content
    assert "ppo_std_streak" in content
    assert "ppo_next_log_std" in content
    assert "ppo_std_cooldown_remaining" in content
    assert "ppo_std_phase" in content
    assert "ppo_std_mean" in content
    assert "ppo_log_std_mean" in content


def test_save_training_curves_writes_parallel_panels(tmp_path: Path) -> None:
    history = [
        {
            "progress": 20,
            "batch_reward_mean": 0.1,
            "episodes_in_window": 32,
            "success_episodes": 16,
            "success_rate": 0.5,
            "det_success_rate": 0.25,
            "det_mean_min_goal_distance": 0.01,
            "approx_kl": 0.02,
            "explained_variance": 0.99,
            "ppo_std_mean": 0.03,
            "ppo_log_std_mean": -3.5,
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "lr": 3e-4,
        }
    ]

    save_training_curves(tmp_path, history)
    html = (tmp_path / "training_curves.html").read_text(encoding="utf-8")

    assert "Batch Reward" in html
    assert "Episodes" in html
    assert "Success Rate" in html
    assert "Deterministic Distance" in html
    assert "Loss" in html
    assert "Approx KL" in html
    assert "Explained Variance" in html
    assert "PPO Std" in html
    assert "Learning Rate" in html
