"""
Datenmodelle für Lernszenarien und Tasks
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class DifficultyLevel(Enum):
    """Schwierigkeitsgrade für Tasks"""

    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4


class TaskType(Enum):
    """Verschiedene Aufgabentypen"""

    MULTIPLE_CHOICE = "multiple_choice"
    FREE_TEXT = "free_text"
    CODE_COMPLETION = "code_completion"
    TRANSLATION = "translation"
    FILL_IN_BLANK = "fill_in_blank"


class Task(BaseModel):
    """Einzelne Lernaufgabe"""

    id: str
    question: str
    answer: str
    task_type: TaskType
    difficulty: DifficultyLevel
    hints: List[str] = Field(default_factory=list)
    category: str = ""
    prerequisites: List[str] = Field(default_factory=list)
    time_limit: Optional[int] = None  # in Sekunden
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LearningScenario(BaseModel):
    """Komplettes Lernszenario mit allen Tasks"""

    name: str
    description: str
    version: str = "1.0"
    tasks: List[Task]
    categories: List[str] = Field(default_factory=list)
    progression_rules: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_tasks_by_difficulty(self, difficulty: DifficultyLevel) -> List[Task]:
        """Gibt alle Tasks mit bestimmtem Schwierigkeitsgrad zurück"""
        return [task for task in self.tasks if task.difficulty == difficulty]

    def get_tasks_by_category(self, category: str) -> List[Task]:
        """Gibt alle Tasks einer bestimmten Kategorie zurück"""
        return [task for task in self.tasks if task.category == category]

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Gibt eine Task anhand ihrer ID zurück"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None


class LearnerState(BaseModel):
    """Zustand des Lernenden"""

    learner_id: str
    scenario_name: str

    # Fortschrittsdaten
    correct_answers: Dict[str, int] = Field(default_factory=dict)  # category -> count
    incorrect_answers: Dict[str, int] = Field(default_factory=dict)  # category -> count
    hints_used: Dict[str, int] = Field(default_factory=dict)  # category -> count

    # Zeitbasierte Metriken
    average_response_times: Dict[str, float] = Field(
        default_factory=dict
    )  # category -> avg_time
    last_activity: Optional[str] = None  # ISO timestamp

    # Aktuelle Session
    current_task_id: Optional[str] = None
    consecutive_correct: int = 0
    consecutive_incorrect: int = 0

    # Geschätzte Fähigkeiten (basierend auf Item Response Theory)
    estimated_ability: Dict[str, float] = Field(
        default_factory=dict
    )  # category -> ability_score

    # Zusätzliche Metadaten
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_success_rate(self, category: str) -> float:
        """Berechnet die Erfolgsrate für eine Kategorie"""
        correct = self.correct_answers.get(category, 0)
        incorrect = self.incorrect_answers.get(category, 0)
        total = correct + incorrect
        return correct / total if total > 0 else 0.0

    def get_total_attempts(self, category: str) -> int:
        """Gibt die Gesamtanzahl der Versuche für eine Kategorie zurück"""
        return self.correct_answers.get(category, 0) + self.incorrect_answers.get(
            category, 0
        )


class AgentAction(BaseModel):
    """Aktion des Lernagenten"""

    action_type: (
        str  # "present_task", "give_hint", "provide_feedback", "change_difficulty"
    )
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str
    reasoning: Optional[str] = None  # Warum diese Aktion gewählt wurde


class LearningSession(BaseModel):
    """Eine komplette Lernsession"""

    session_id: str
    learner_id: str
    scenario_name: str
    start_time: str
    end_time: Optional[str] = None
    actions: List[AgentAction] = Field(default_factory=list)
    final_state: Optional[LearnerState] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
