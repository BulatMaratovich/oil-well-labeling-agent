"""
Device profile store backed by SQLite.

Stores per-device operating statistics and labeled examples.
Provides k-NN similarity search to find devices with similar operating profiles,
enabling few-shot generalization to devices with no or few labeled examples.

Scale target: 10,000 devices. SQLite is sufficient for PoC;
production would use a proper vector store or time-series DB.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from src.models.schemas import (
    DeviceProfile,
    DeviceStats,
    LabeledExample,
)


class ProfileStore:
    """
    Persistent store for device profiles and labeled examples.

    Thread-safety: single-connection for PoC. Production should use
    connection pooling or a proper database.
    """

    def __init__(self, db_path: str = "data/profiles.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS device_profiles (
                device_id TEXT PRIMARY KEY,
                op_mean REAL NOT NULL,
                op_std REAL NOT NULL,
                op_p10 REAL NOT NULL,
                op_p90 REAL NOT NULL,
                op_median REAL NOT NULL,
                amp_mean REAL NOT NULL,
                amp_std REAL NOT NULL,
                amp_p10 REAL NOT NULL,
                amp_p90 REAL NOT NULL,
                amp_median REAL NOT NULL,
                idle_mean REAL,
                idle_std REAL,
                stop_mean REAL,
                stop_std REAL,
                sample_count INTEGER DEFAULT 0,
                last_updated TEXT NOT NULL,
                has_labeled_examples INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS labeled_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                scenario_id TEXT NOT NULL,
                label TEXT NOT NULL,
                level_ratio REAL NOT NULL,
                amplitude_ratio REAL NOT NULL,
                window_mean REAL NOT NULL,
                window_std REAL NOT NULL,
                operating_mean REAL NOT NULL,
                description TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_examples_scenario
                ON labeled_examples (scenario_id, label);
            CREATE INDEX IF NOT EXISTS idx_examples_device
                ON labeled_examples (device_id, scenario_id);
        """)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    #  Profile CRUD                                                        #
    # ------------------------------------------------------------------ #

    def get_profile(self, device_id: str) -> Optional[DeviceProfile]:
        row = self.conn.execute(
            "SELECT * FROM device_profiles WHERE device_id = ?", (device_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_profile(row)

    def save_profile(self, profile: DeviceProfile) -> None:
        idle_mean = profile.idle_stats.mean if profile.idle_stats else None
        idle_std = profile.idle_stats.std if profile.idle_stats else None
        stop_mean = profile.stop_stats.mean if profile.stop_stats else None
        stop_std = profile.stop_stats.std if profile.stop_stats else None

        self.conn.execute("""
            INSERT INTO device_profiles (
                device_id, op_mean, op_std, op_p10, op_p90, op_median,
                amp_mean, amp_std, amp_p10, amp_p90, amp_median,
                idle_mean, idle_std, stop_mean, stop_std,
                sample_count, last_updated, has_labeled_examples
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(device_id) DO UPDATE SET
                op_mean=excluded.op_mean, op_std=excluded.op_std,
                op_p10=excluded.op_p10, op_p90=excluded.op_p90,
                op_median=excluded.op_median,
                amp_mean=excluded.amp_mean, amp_std=excluded.amp_std,
                amp_p10=excluded.amp_p10, amp_p90=excluded.amp_p90,
                amp_median=excluded.amp_median,
                idle_mean=excluded.idle_mean, idle_std=excluded.idle_std,
                stop_mean=excluded.stop_mean, stop_std=excluded.stop_std,
                sample_count=excluded.sample_count,
                last_updated=excluded.last_updated,
                has_labeled_examples=excluded.has_labeled_examples
        """, (
            profile.device_id,
            profile.operating_stats.mean, profile.operating_stats.std,
            profile.operating_stats.p10, profile.operating_stats.p90,
            profile.operating_stats.median,
            profile.amplitude_stats.mean, profile.amplitude_stats.std,
            profile.amplitude_stats.p10, profile.amplitude_stats.p90,
            profile.amplitude_stats.median,
            idle_mean, idle_std, stop_mean, stop_std,
            profile.sample_count,
            profile.last_updated.isoformat(),
            int(profile.has_labeled_examples),
        ))
        self.conn.commit()

    def update_profile_samples(
        self,
        device_id: str,
        new_samples: list[float],
        profile: Optional[DeviceProfile] = None,
    ) -> DeviceProfile:
        """
        Incrementally update a device profile with new normal-operation samples.

        Uses exponential moving average to adapt to gradual parameter drift
        (e.g., motor aging, belt wear) while ignoring transient anomalies.
        """
        from src.tools.signal_processing import build_profile_from_samples

        if profile is None:
            profile = self.get_profile(device_id)

        new_stats = build_profile_from_samples(
            device_id=device_id,
            timestamps=[datetime.utcnow()] * len(new_samples),  # placeholder
            values=new_samples,
        )

        if profile is None or profile.sample_count < 50:
            # Cold start: use new samples directly
            updated = DeviceProfile(
                device_id=device_id,
                operating_stats=new_stats,
                amplitude_stats=DeviceStats(
                    mean=float(np.std(new_samples)),
                    std=0.0, p10=0.0, p90=0.0, median=0.0,
                ),
                sample_count=len(new_samples),
                last_updated=datetime.utcnow(),
            )
        else:
            # Warm update: exponential moving average (alpha=0.2)
            alpha = 0.2
            op = profile.operating_stats
            updated_op = DeviceStats(
                mean=alpha * new_stats.mean + (1 - alpha) * op.mean,
                std=alpha * new_stats.std + (1 - alpha) * op.std,
                p10=alpha * new_stats.p10 + (1 - alpha) * op.p10,
                p90=alpha * new_stats.p90 + (1 - alpha) * op.p90,
                median=alpha * new_stats.median + (1 - alpha) * op.median,
            )
            updated = DeviceProfile(
                device_id=device_id,
                operating_stats=updated_op,
                amplitude_stats=profile.amplitude_stats,
                idle_stats=profile.idle_stats,
                stop_stats=profile.stop_stats,
                sample_count=profile.sample_count + len(new_samples),
                last_updated=datetime.utcnow(),
                has_labeled_examples=profile.has_labeled_examples,
            )

        self.save_profile(updated)
        return updated

    # ------------------------------------------------------------------ #
    #  Similarity Search                                                   #
    # ------------------------------------------------------------------ #

    def get_similar_devices(
        self,
        profile: DeviceProfile,
        k: int = 5,
        scenario_id: Optional[str] = None,
    ) -> list[str]:
        """
        Find k most similar devices by operating profile.

        Similarity is computed in a 3D space:
          (log(operating_mean), operating_std/operating_mean, amplitude_ratio)

        Using log(mean) because motor power spans 3 orders of magnitude.
        Using CV (std/mean) instead of absolute std for scale-invariance.

        If scenario_id is provided, filters to devices with labeled examples
        for that scenario (for few-shot retrieval).
        """
        if scenario_id:
            rows = self.conn.execute("""
                SELECT dp.* FROM device_profiles dp
                INNER JOIN labeled_examples le ON le.device_id = dp.device_id
                WHERE dp.device_id != ? AND le.scenario_id = ?
                GROUP BY dp.device_id
            """, (profile.device_id, scenario_id)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM device_profiles WHERE device_id != ?",
                (profile.device_id,)
            ).fetchall()

        if not rows:
            return []

        # Feature vector for target device
        target = self._profile_to_feature_vector(profile)

        # Compute cosine similarity to all candidates
        scores = []
        for row in rows:
            candidate = self._row_to_profile(row)
            vec = self._profile_to_feature_vector(candidate)
            sim = self._cosine_similarity(target, vec)
            scores.append((candidate.device_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [device_id for device_id, _ in scores[:k]]

    def _profile_to_feature_vector(self, profile: DeviceProfile) -> np.ndarray:
        """Convert profile to normalized feature vector for similarity computation."""
        op = profile.operating_stats
        amp = profile.amplitude_stats

        log_mean = np.log1p(abs(op.mean))  # log scale for wide power range
        cv = op.std / max(abs(op.mean), 1e-6)  # coefficient of variation
        amp_ratio = amp.mean / max(abs(op.mean), 1e-6)  # amplitude relative to scale

        return np.array([log_mean, cv, amp_ratio], dtype=float)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # ------------------------------------------------------------------ #
    #  Labeled Examples                                                    #
    # ------------------------------------------------------------------ #

    def add_labeled_example(
        self,
        device_id: str,
        scenario_id: str,
        label: str,
        level_ratio: float,
        amplitude_ratio: float,
        window_mean: float,
        window_std: float,
        operating_mean: float,
        description: str,
    ) -> None:
        self.conn.execute("""
            INSERT INTO labeled_examples (
                device_id, scenario_id, label, level_ratio, amplitude_ratio,
                window_mean, window_std, operating_mean, description, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            device_id, scenario_id, label, level_ratio, amplitude_ratio,
            window_mean, window_std, operating_mean, description,
            datetime.utcnow().isoformat(),
        ))
        # Mark device as having labeled examples
        self.conn.execute(
            "UPDATE device_profiles SET has_labeled_examples=1 WHERE device_id=?",
            (device_id,)
        )
        self.conn.commit()

    def get_examples_for_devices(
        self,
        device_ids: list[str],
        scenario_id: str,
        max_per_device: int = 2,
    ) -> list[LabeledExample]:
        """Retrieve labeled examples for a set of similar devices."""
        if not device_ids:
            return []

        placeholders = ",".join("?" * len(device_ids))
        rows = self.conn.execute(f"""
            SELECT * FROM labeled_examples
            WHERE device_id IN ({placeholders}) AND scenario_id = ?
            ORDER BY device_id, created_at DESC
        """, (*device_ids, scenario_id)).fetchall()

        # Limit to max_per_device per device
        examples = []
        counts: dict[str, int] = {}
        for row in rows:
            did = row["device_id"]
            if counts.get(did, 0) >= max_per_device:
                continue
            counts[did] = counts.get(did, 0) + 1
            examples.append(LabeledExample(
                device_id=did,
                label=row["label"],
                level_ratio=row["level_ratio"],
                amplitude_ratio=row["amplitude_ratio"],
                window_mean=row["window_mean"],
                window_std=row["window_std"],
                operating_mean=row["operating_mean"],
                description=row["description"],
            ))

        return examples

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _row_to_profile(self, row: sqlite3.Row) -> DeviceProfile:
        idle_stats = None
        if row["idle_mean"] is not None:
            idle_stats = DeviceStats(
                mean=row["idle_mean"], std=row["idle_std"] or 0.0,
                p10=0.0, p90=0.0, median=row["idle_mean"],
            )
        stop_stats = None
        if row["stop_mean"] is not None:
            stop_stats = DeviceStats(
                mean=row["stop_mean"], std=row["stop_std"] or 0.0,
                p10=0.0, p90=0.0, median=row["stop_mean"],
            )
        return DeviceProfile(
            device_id=row["device_id"],
            operating_stats=DeviceStats(
                mean=row["op_mean"], std=row["op_std"],
                p10=row["op_p10"], p90=row["op_p90"], median=row["op_median"],
            ),
            amplitude_stats=DeviceStats(
                mean=row["amp_mean"], std=row["amp_std"],
                p10=row["amp_p10"], p90=row["amp_p90"], median=row["amp_median"],
            ),
            idle_stats=idle_stats,
            stop_stats=stop_stats,
            sample_count=row["sample_count"],
            last_updated=datetime.fromisoformat(row["last_updated"]),
            has_labeled_examples=bool(row["has_labeled_examples"]),
        )

    def get_all_device_ids(self) -> list[str]:
        rows = self.conn.execute("SELECT device_id FROM device_profiles").fetchall()
        return [r["device_id"] for r in rows]

    def close(self) -> None:
        self.conn.close()
