"""
Automatisiertes Training System für RL-Agenten
Trainiert Agenten automatisch basierend auf Szenarien und Performance-Metriken
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import json
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
import logging

from .agent import LearningAgent
from .scenario_manager import ScenarioManager
from .models import LearningScenario


class TrainingStatus(Enum):
    """Status des Trainings"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class TrainingPriority(Enum):
    """Priorität des Trainings"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TrainingJob:
    """Repräsentiert einen Training-Job"""

    id: str
    scenario_name: str
    algorithm: str = "DQN"
    timesteps: int = 10000
    priority: TrainingPriority = TrainingPriority.NORMAL
    status: TrainingStatus = TrainingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    auto_retrain_threshold: float = 0.6  # Retrain wenn Performance unter diesem Wert
    max_retries: int = 3
    retry_count: int = 0

    @property
    def duration(self) -> Optional[timedelta]:
        """Berechnet die Trainingsdauer"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return datetime.now() - self.started_at
        return None

    @property
    def is_finished(self) -> bool:
        """Prüft ob der Job beendet ist"""
        return self.status in [
            TrainingStatus.COMPLETED,
            TrainingStatus.FAILED,
            TrainingStatus.CANCELLED,
        ]


class AutoTrainingManager:
    """Verwaltet automatisiertes Training für RL-Agenten"""

    def __init__(
        self,
        models_dir: str = "models/",
        logs_dir: str = "logs/training/",
        max_concurrent_jobs: int = 2,
    ):
        self.models_dir = Path(models_dir)
        self.logs_dir = Path(logs_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.scenario_manager = ScenarioManager()
        self.max_concurrent_jobs = max_concurrent_jobs

        # Job Management
        self.job_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.running_jobs: Dict[str, TrainingJob] = {}
        self.completed_jobs: Dict[str, TrainingJob] = {}
        self.job_history: List[TrainingJob] = []

        # Threading
        self.worker_threads: List[threading.Thread] = []
        self.is_running = False
        self.shutdown_event = threading.Event()

        # Callbacks
        self.progress_callbacks: List[Callable[[str, float], None]] = []
        self.completion_callbacks: List[Callable[[TrainingJob], None]] = []

        # Setup Logging
        self.setup_logging()

        # Load existing jobs
        self.load_job_history()

    def setup_logging(self):
        """Konfiguriert Logging für das Training"""
        log_file = self.logs_dir / "auto_training.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def add_training_job(
        self,
        scenario_name: str,
        algorithm: str = "DQN",
        timesteps: int = 10000,
        priority: TrainingPriority = TrainingPriority.NORMAL,
        auto_retrain_threshold: float = 0.6,
    ) -> str:
        """
        Fügt einen neuen Training-Job zur Warteschlange hinzu

        Returns:
            Job-ID
        """
        job_id = f"job_{scenario_name}_{algorithm}_{int(time.time())}"

        job = TrainingJob(
            id=job_id,
            scenario_name=scenario_name,
            algorithm=algorithm,
            timesteps=timesteps,
            priority=priority,
            auto_retrain_threshold=auto_retrain_threshold,
        )

        # Zur Warteschlange hinzufügen (negative Priorität für korrekte Sortierung)
        self.job_queue.put((-priority.value, time.time(), job))

        self.logger.info(
            f"Training-Job hinzugefügt: {job_id} für Szenario {scenario_name}"
        )
        return job_id

    def start_auto_training(self):
        """Startet das automatische Training-System"""
        if self.is_running:
            self.logger.warning("Training-System läuft bereits")
            return

        self.is_running = True
        self.shutdown_event.clear()

        # Starte Worker-Threads
        for i in range(self.max_concurrent_jobs):
            worker = threading.Thread(
                target=self._training_worker, args=(i,), daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)

        # Starte Monitoring-Thread
        monitor = threading.Thread(target=self._monitoring_worker, daemon=True)
        monitor.start()
        self.worker_threads.append(monitor)

        self.logger.info(
            f"Auto-Training gestartet mit {self.max_concurrent_jobs} Worker-Threads"
        )

    def stop_auto_training(self):
        """Stoppt das automatische Training-System"""
        if not self.is_running:
            return

        self.logger.info("Stoppe Auto-Training...")
        self.is_running = False
        self.shutdown_event.set()

        # Warte auf alle Threads
        for thread in self.worker_threads:
            thread.join(timeout=10)

        self.worker_threads.clear()
        self.logger.info("Auto-Training gestoppt")

    def _training_worker(self, worker_id: int):
        """Worker-Thread für das Training"""
        self.logger.info(f"Training-Worker {worker_id} gestartet")

        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Hole nächsten Job aus der Warteschlange (mit Timeout)
                try:
                    _, _, job = self.job_queue.get(timeout=5)
                except queue.Empty:
                    continue

                # Führe Training aus
                self._execute_training_job(job, worker_id)

            except Exception as e:
                self.logger.error(f"Fehler in Worker {worker_id}: {e}")

        self.logger.info(f"Training-Worker {worker_id} beendet")

    def _execute_training_job(self, job: TrainingJob, worker_id: int):
        """Führt einen einzelnen Training-Job aus"""
        try:
            self.logger.info(f"Worker {worker_id}: Starte Training für Job {job.id}")

            # Job Status aktualisieren
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now()
            self.running_jobs[job.id] = job

            # Szenario laden
            scenario = self.scenario_manager.load_scenario(job.scenario_name)
            if not scenario:
                raise Exception(f"Szenario {job.scenario_name} nicht gefunden")

            # Agent erstellen
            agent = LearningAgent(scenario, algorithm=job.algorithm)

            # Progress Callback definieren
            def progress_callback(progress: float):
                job.progress = progress
                for callback in self.progress_callbacks:
                    callback(job.id, progress)

            # Training durchführen
            episode_rewards, episode_lengths = agent.train(
                total_timesteps=job.timesteps,
                save_path=str(self.models_dir),
                progress_callback=progress_callback,
            )

            # Evaluation
            eval_results = agent.evaluate(n_episodes=10)

            # Ergebnisse speichern
            job.results = {
                "mean_reward": eval_results["mean_reward"],
                "success_rate": eval_results["mean_success_rate"],
                "episodes": len(episode_rewards),
                "final_episode_reward": episode_rewards[-1] if episode_rewards else 0,
                "convergence_episode": self._find_convergence_point(episode_rewards),
            }

            # Job als erfolgreich markieren
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            job.progress = 1.0

            self.logger.info(
                f"Training abgeschlossen für Job {job.id}: "
                f"Success Rate: {eval_results['mean_success_rate']:.2%}"
            )

            # Prüfe ob Nachtraining nötig ist
            if eval_results["mean_success_rate"] < job.auto_retrain_threshold:
                self._schedule_retrain(job)

        except Exception as e:
            self.logger.error(f"Training fehlgeschlagen für Job {job.id}: {e}")
            job.status = TrainingStatus.FAILED
            job.completed_at = datetime.now()
            job.error_message = str(e)

        finally:
            # Job aus running_jobs entfernen und zu completed_jobs hinzufügen
            if job.id in self.running_jobs:
                del self.running_jobs[job.id]

            self.completed_jobs[job.id] = job
            self.job_history.append(job)

            # Completion Callbacks aufrufen
            for callback in self.completion_callbacks:
                callback(job)

            # Job History speichern
            self.save_job_history()

    def _find_convergence_point(
        self, episode_rewards: List[float], window_size: int = 50
    ) -> Optional[int]:
        """Findet den Konvergenzpunkt im Training"""
        if len(episode_rewards) < window_size * 2:
            return None

        # Berechne gleitenden Durchschnitt
        moving_avg = []
        for i in range(window_size, len(episode_rewards)):
            avg = sum(episode_rewards[i - window_size : i]) / window_size
            moving_avg.append(avg)

        # Finde Punkt wo Verbesserung minimal wird
        for i in range(window_size, len(moving_avg)):
            recent_avg = sum(moving_avg[i - window_size : i]) / window_size
            if i > 0:
                prev_avg = (
                    sum(moving_avg[max(0, i - window_size * 2) : i - window_size])
                    / window_size
                )
                improvement = (
                    (recent_avg - prev_avg) / abs(prev_avg) if prev_avg != 0 else 0
                )

                if improvement < 0.05:  # Weniger als 5% Verbesserung
                    return i + window_size

        return None

    def _schedule_retrain(self, original_job: TrainingJob):
        """Plant ein Nachtraining wenn Performance unzureichend ist"""
        if original_job.retry_count >= original_job.max_retries:
            self.logger.warning(f"Max Retries erreicht für Job {original_job.id}")
            return

        # Erhöhe Timesteps für Nachtraining
        new_timesteps = int(original_job.timesteps * 1.5)

        retrain_job_id = self.add_training_job(
            scenario_name=original_job.scenario_name,
            algorithm=original_job.algorithm,
            timesteps=new_timesteps,
            priority=TrainingPriority.HIGH,
            auto_retrain_threshold=original_job.auto_retrain_threshold,
        )

        # Update retry count
        if retrain_job_id in [job.id for _, _, job in list(self.job_queue.queue)]:
            for _, _, job in list(self.job_queue.queue):
                if job.id == retrain_job_id:
                    job.retry_count = original_job.retry_count + 1
                    break

        self.logger.info(
            f"Nachtraining geplant: {retrain_job_id} (Versuch {original_job.retry_count + 1})"
        )

    def _monitoring_worker(self):
        """Überwacht das System und führt regelmäßige Wartung durch"""
        self.logger.info("Monitoring-Worker gestartet")

        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Überwache laufende Jobs
                for job_id, job in list(self.running_jobs.items()):
                    if job.duration and job.duration > timedelta(
                        hours=6
                    ):  # Timeout nach 6 Stunden
                        self.logger.warning(
                            f"Job {job_id} läuft bereits {job.duration} - möglicher Deadlock"
                        )

                # Performance-basiertes automatisches Training
                self._check_performance_degradation()

                # Cleanup alte Jobs
                self._cleanup_old_jobs()

                # Warte 5 Minuten bis zum nächsten Check
                self.shutdown_event.wait(300)

            except Exception as e:
                self.logger.error(f"Fehler im Monitoring: {e}")

        self.logger.info("Monitoring-Worker beendet")

    def _check_performance_degradation(self):
        """Prüft auf Performance-Verschlechterung und startet ggf. Nachtraining"""
        # TODO: Implementiere Performance-Monitoring
        # Könnte z.B. User-Session-Daten analysieren
        pass

    def _cleanup_old_jobs(self, max_age_days: int = 30):
        """Bereinigt alte Job-Einträge"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        # Entferne alte Jobs aus completed_jobs
        to_remove = [
            job_id
            for job_id, job in self.completed_jobs.items()
            if job.completed_at and job.completed_at < cutoff_date
        ]

        for job_id in to_remove:
            del self.completed_jobs[job_id]

        if to_remove:
            self.logger.info(f"Bereinigt {len(to_remove)} alte Jobs")

    def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Gibt Status eines Jobs zurück"""
        # Suche in laufenden Jobs
        if job_id in self.running_jobs:
            return self.running_jobs[job_id]

        # Suche in abgeschlossenen Jobs
        if job_id in self.completed_jobs:
            return self.completed_jobs[job_id]

        # Suche in Warteschlange
        for _, _, job in list(self.job_queue.queue):
            if job.id == job_id:
                return job

        return None

    def get_system_stats(self) -> Dict[str, Any]:
        """Gibt System-Statistiken zurück"""
        queue_size = self.job_queue.qsize()
        running_count = len(self.running_jobs)
        completed_count = len(self.completed_jobs)

        # Erfolgsrate berechnen
        completed_jobs = [
            job for job in self.job_history if job.status == TrainingStatus.COMPLETED
        ]
        success_rate = (
            len(completed_jobs) / len(self.job_history) if self.job_history else 0
        )

        # Durchschnittliche Trainingszeit
        durations = [
            job.duration.total_seconds() for job in completed_jobs if job.duration
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "queue_size": queue_size,
            "running_jobs": running_count,
            "completed_jobs": completed_count,
            "total_jobs": len(self.job_history),
            "success_rate": success_rate,
            "average_duration_seconds": avg_duration,
            "is_running": self.is_running,
            "max_concurrent_jobs": self.max_concurrent_jobs,
        }

    def cancel_job(self, job_id: str) -> bool:
        """Bricht einen Job ab"""
        job = self.get_job_status(job_id)
        if not job:
            return False

        if job.status == TrainingStatus.PENDING:
            # Entferne aus Warteschlange
            job.status = TrainingStatus.CANCELLED
            return True
        elif job.status == TrainingStatus.RUNNING:
            # TODO: Implementiere Abbruch laufender Jobs
            self.logger.warning(
                f"Abbruch laufender Jobs noch nicht implementiert: {job_id}"
            )
            return False

        return False

    def pause_job(self, job_id: str) -> bool:
        """Pausiert einen Job"""
        # TODO: Implementiere Job-Pausierung
        self.logger.warning("Job-Pausierung noch nicht implementiert")
        return False

    def add_progress_callback(self, callback: Callable[[str, float], None]):
        """Fügt Progress-Callback hinzu"""
        self.progress_callbacks.append(callback)

    def add_completion_callback(self, callback: Callable[[TrainingJob], None]):
        """Fügt Completion-Callback hinzu"""
        self.completion_callbacks.append(callback)

    def save_job_history(self):
        """Speichert Job-History in JSON-Datei"""
        history_file = self.logs_dir / "job_history.json"

        history_data = []
        for job in self.job_history:
            job_data = {
                "id": job.id,
                "scenario_name": job.scenario_name,
                "algorithm": job.algorithm,
                "timesteps": job.timesteps,
                "priority": job.priority.name,
                "status": job.status.name,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat()
                if job.completed_at
                else None,
                "progress": job.progress,
                "results": job.results,
                "error_message": job.error_message,
                "retry_count": job.retry_count,
            }
            history_data.append(job_data)

        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)

    def load_job_history(self):
        """Lädt Job-History aus JSON-Datei"""
        history_file = self.logs_dir / "job_history.json"

        if not history_file.exists():
            return

        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history_data = json.load(f)

            for job_data in history_data:
                job = TrainingJob(
                    id=job_data["id"],
                    scenario_name=job_data["scenario_name"],
                    algorithm=job_data["algorithm"],
                    timesteps=job_data["timesteps"],
                    priority=TrainingPriority[job_data["priority"]],
                    status=TrainingStatus[job_data["status"]],
                    created_at=datetime.fromisoformat(job_data["created_at"]),
                    started_at=datetime.fromisoformat(job_data["started_at"])
                    if job_data["started_at"]
                    else None,
                    completed_at=datetime.fromisoformat(job_data["completed_at"])
                    if job_data["completed_at"]
                    else None,
                    progress=job_data["progress"],
                    results=job_data["results"],
                    error_message=job_data["error_message"],
                    retry_count=job_data["retry_count"],
                )

                self.job_history.append(job)

                # Füge zu completed_jobs hinzu wenn abgeschlossen
                if job.is_finished:
                    self.completed_jobs[job.id] = job

        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Job-History: {e}")


# Convenience-Funktionen
def schedule_all_scenarios_training(
    auto_trainer: AutoTrainingManager, algorithm: str = "DQN", timesteps: int = 10000
) -> List[str]:
    """Plant Training für alle verfügbaren Szenarien"""
    scenario_manager = ScenarioManager()
    scenarios = scenario_manager.list_scenarios()

    job_ids = []
    for scenario_name in scenarios:
        job_id = auto_trainer.add_training_job(
            scenario_name=scenario_name,
            algorithm=algorithm,
            timesteps=timesteps,
            priority=TrainingPriority.NORMAL,
        )
        job_ids.append(job_id)

    return job_ids


def create_training_schedule(
    scenarios: List[str], algorithms: List[str] = None
) -> List[Dict[str, Any]]:
    """Erstellt einen Trainingsplan für mehrere Szenarien und Algorithmen"""
    if algorithms is None:
        algorithms = ["DQN", "A2C", "PPO"]

    schedule = []
    for scenario in scenarios:
        for algorithm in algorithms:
            schedule.append(
                {
                    "scenario_name": scenario,
                    "algorithm": algorithm,
                    "timesteps": 10000,
                    "priority": TrainingPriority.NORMAL,
                }
            )

    return schedule
