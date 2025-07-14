"""
Spaced Repetition Algorithm für zeitbasierte Wiederholung von Aufgaben
Implementiert eine Variante des SM-2 Algorithmus (SuperMemo 2)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from .models import Task


class ReviewQuality(Enum):
    """Bewertungsqualität für Spaced Repetition"""

    BLACKOUT = 0  # Kompletter Blackout
    INCORRECT = 1  # Inkorrekt, aber erinnert
    HARD = 2  # Korrekt mit großer Schwierigkeit
    GOOD = 3  # Korrekt mit etwas Schwierigkeit
    EASY = 4  # Korrekt ohne Schwierigkeit
    PERFECT = 5  # Perfekt und schnell


@dataclass
class SpacedRepetitionCard:
    """Repräsentiert eine Karte im Spaced Repetition System"""

    task_id: str
    easiness_factor: float = 2.5  # Einfachheitsfaktor (1.3 - 2.5+)
    interval: int = 1  # Tage bis zur nächsten Wiederholung
    repetitions: int = 0  # Anzahl erfolgreicher Wiederholungen
    next_review: datetime = field(default_factory=datetime.now)
    last_review: Optional[datetime] = None
    review_history: List[Tuple[datetime, int, int]] = field(
        default_factory=list
    )  # (datum, qualität, interval)

    def is_due(self) -> bool:
        """Prüft ob die Karte zur Wiederholung fällig ist"""
        return datetime.now() >= self.next_review

    def days_overdue(self) -> int:
        """Anzahl Tage überfällig"""
        if not self.is_due():
            return 0
        return (datetime.now() - self.next_review).days

    def update_scheduling(self, quality: ReviewQuality) -> None:
        """
        Aktualisiert das Scheduling basierend auf der Bewertungsqualität
        Implementiert den SM-2 Algorithmus
        """
        quality_score = quality.value

        # Speichere Review in History
        self.review_history.append((datetime.now(), quality_score, self.interval))
        self.last_review = datetime.now()

        # SM-2 Algorithmus
        if quality_score >= 3:  # Korrekte Antwort
            if self.repetitions == 0:
                self.interval = 1
            elif self.repetitions == 1:
                self.interval = 6
            else:
                self.interval = round(self.interval * self.easiness_factor)

            self.repetitions += 1
        else:  # Inkorrekte Antwort
            self.repetitions = 0
            self.interval = 0  # Sofort wieder fällig bei falscher Antwort

        # Easiness Factor anpassen
        self.easiness_factor = max(
            1.3,
            self.easiness_factor
            + (0.1 - (5 - quality_score) * (0.08 + (5 - quality_score) * 0.02)),
        )

        # Nächstes Review-Datum setzen
        if self.interval == 0:
            # Bei falschen Antworten: sofort wieder fällig
            self.next_review = datetime.now()
        else:
            # Für normale Reviews: Setze das Datum auf den nächsten Tag um Mitternacht
            # Dies vermeidet Verwirrung bei der Tagesberechnung
            from datetime import date

            today = date.today()
            next_review_date = today + timedelta(days=self.interval)
            # Setze auf Beginn des Tages (00:00:00)
            self.next_review = datetime.combine(next_review_date, datetime.min.time())


class SpacedRepetitionManager:
    """Verwaltet das Spaced Repetition System für alle Tasks"""

    def __init__(self, data_dir: str = "data/spaced_repetition/"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cards: Dict[str, SpacedRepetitionCard] = {}
        self.load_cards()

    def add_task(self, task: Task, user_id: str = "default") -> None:
        """Fügt eine neue Task zum Spaced Repetition System hinzu"""
        card_id = f"{user_id}_{task.id}"
        if card_id not in self.cards:
            self.cards[card_id] = SpacedRepetitionCard(
                task_id=task.id,
                next_review=datetime.now(),  # Sofort verfügbar für erste Wiederholung
            )
            self.save_cards()

    def get_due_cards(
        self, user_id: str = "default", limit: int = None
    ) -> List[SpacedRepetitionCard]:
        """Gibt alle fälligen Karten zurück, sortiert nach Priorität"""
        user_cards = [
            card
            for card_id, card in self.cards.items()
            if card_id.startswith(f"{user_id}_")
        ]
        due_cards = [card for card in user_cards if card.is_due()]

        # Sortiere nach Priorität: überfällige zuerst, dann nach Schwierigkeit
        due_cards.sort(
            key=lambda c: (c.days_overdue(), -c.easiness_factor), reverse=True
        )

        return due_cards[:limit] if limit else due_cards

    def record_review(
        self, task_id: str, quality: ReviewQuality, user_id: str = "default"
    ) -> None:
        """Zeichnet eine Bewertung auf und aktualisiert das Scheduling"""
        card_id = f"{user_id}_{task_id}"

        if card_id in self.cards:
            self.cards[card_id].update_scheduling(quality)
            self.save_cards()
        else:
            print(f"Warnung: Karte {card_id} nicht gefunden")

    def get_card_stats(self, user_id: str = "default") -> Dict[str, int]:
        """Gibt Statistiken über die Karten zurück"""
        user_cards = [
            card
            for card_id, card in self.cards.items()
            if card_id.startswith(f"{user_id}_")
        ]

        due_count = len([card for card in user_cards if card.is_due()])
        total_count = len(user_cards)
        mature_count = len([card for card in user_cards if card.repetitions >= 3])

        return {
            "total": total_count,
            "due": due_count,
            "mature": mature_count,
            "learning": total_count - mature_count,
        }

    def get_next_review_schedule(
        self, user_id: str = "default", days: int = 7
    ) -> Dict[str, int]:
        """Gibt die Anzahl Reviews für die nächsten X Tage zurück"""
        user_cards = [
            card
            for card_id, card in self.cards.items()
            if card_id.startswith(f"{user_id}_")
        ]
        schedule = {}

        today = datetime.now().date()
        for i in range(days):
            date = today + timedelta(days=i)
            count = len(
                [card for card in user_cards if card.next_review.date() == date]
            )
            schedule[date.strftime("%Y-%m-%d")] = count

        return schedule

    def get_retention_stats(
        self, user_id: str = "default", days: int = 30
    ) -> Dict[str, float]:
        """Berechnet Retentionsstatistiken"""
        user_cards = [
            card
            for card_id, card in self.cards.items()
            if card_id.startswith(f"{user_id}_")
        ]

        # Filtere Reviews der letzten X Tage
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_reviews = []

        for card in user_cards:
            for review_date, quality, interval in card.review_history:
                if review_date >= cutoff_date:
                    recent_reviews.append((review_date, quality, interval))

        if not recent_reviews:
            return {"retention_rate": 0.0, "average_quality": 0.0, "total_reviews": 0}

        # Berechne Statistiken
        correct_reviews = len([r for r in recent_reviews if r[1] >= 3])
        total_reviews = len(recent_reviews)
        retention_rate = correct_reviews / total_reviews if total_reviews > 0 else 0
        average_quality = sum(r[1] for r in recent_reviews) / total_reviews

        return {
            "retention_rate": retention_rate,
            "average_quality": average_quality,
            "total_reviews": total_reviews,
            "correct_reviews": correct_reviews,
        }

    def suggest_optimal_timing(self, user_id: str = "default") -> Dict[str, Any]:
        """Schlägt optimale Lernzeiten vor basierend auf Performance"""
        stats = self.get_retention_stats(user_id)
        due_cards = self.get_due_cards(user_id)

        # Einfache Heuristik für optimales Timing
        if stats["retention_rate"] > 0.9:
            suggested_daily_limit = min(50, len(due_cards))
            difficulty = "low"
        elif stats["retention_rate"] > 0.7:
            suggested_daily_limit = min(30, len(due_cards))
            difficulty = "medium"
        else:
            suggested_daily_limit = min(20, len(due_cards))
            difficulty = "high"

        return {
            "suggested_daily_limit": suggested_daily_limit,
            "difficulty_assessment": difficulty,
            "due_cards_count": len(due_cards),
            "retention_rate": stats["retention_rate"],
        }

    def save_cards(self) -> None:
        """Speichert alle Karten in JSON-Datei"""
        cards_data = {}
        for card_id, card in self.cards.items():
            cards_data[card_id] = {
                "task_id": card.task_id,
                "easiness_factor": card.easiness_factor,
                "interval": card.interval,
                "repetitions": card.repetitions,
                "next_review": card.next_review.isoformat(),
                "last_review": card.last_review.isoformat()
                if card.last_review
                else None,
                "review_history": [
                    (r[0].isoformat(), r[1], r[2]) for r in card.review_history
                ],
            }

        cards_file = self.data_dir / "spaced_repetition_cards.json"
        with open(cards_file, "w", encoding="utf-8") as f:
            json.dump(cards_data, f, indent=2, ensure_ascii=False)

    def load_cards(self) -> None:
        """Lädt alle Karten aus JSON-Datei"""
        cards_file = self.data_dir / "spaced_repetition_cards.json"

        if not cards_file.exists():
            return

        try:
            with open(cards_file, "r", encoding="utf-8") as f:
                cards_data = json.load(f)

            for card_id, data in cards_data.items():
                card = SpacedRepetitionCard(
                    task_id=data["task_id"],
                    easiness_factor=data["easiness_factor"],
                    interval=data["interval"],
                    repetitions=data["repetitions"],
                    next_review=datetime.fromisoformat(data["next_review"]),
                    last_review=datetime.fromisoformat(data["last_review"])
                    if data["last_review"]
                    else None,
                    review_history=[
                        (datetime.fromisoformat(r[0]), r[1], r[2])
                        for r in data["review_history"]
                    ],
                )
                self.cards[card_id] = card

        except Exception as e:
            print(f"Fehler beim Laden der Spaced Repetition Karten: {e}")

    def reset_card(self, task_id: str, user_id: str = "default") -> None:
        """Setzt eine Karte zurück (z.B. bei Inhaltsaktualisierung)"""
        card_id = f"{user_id}_{task_id}"
        if card_id in self.cards:
            self.cards[card_id] = SpacedRepetitionCard(
                task_id=task_id, next_review=datetime.now()
            )
            self.save_cards()

    def export_progress(self, user_id: str = "default") -> Dict[str, Any]:
        """Exportiert Lernfortschritt für Analyse"""
        user_cards = [
            card
            for card_id, card in self.cards.items()
            if card_id.startswith(f"{user_id}_")
        ]

        export_data = {
            "user_id": user_id,
            "export_date": datetime.now().isoformat(),
            "statistics": self.get_card_stats(user_id),
            "retention_stats": self.get_retention_stats(user_id),
            "cards": [],
        }

        for card in user_cards:
            card_data = {
                "task_id": card.task_id,
                "easiness_factor": card.easiness_factor,
                "interval": card.interval,
                "repetitions": card.repetitions,
                "next_review": card.next_review.isoformat(),
                "last_review": card.last_review.isoformat()
                if card.last_review
                else None,
                "review_count": len(card.review_history),
                "average_quality": sum(r[1] for r in card.review_history)
                / len(card.review_history)
                if card.review_history
                else 0,
            }
            export_data["cards"].append(card_data)

        return export_data


def determine_review_quality(
    is_correct: bool, response_time: float, hints_used: int
) -> ReviewQuality:
    """
    Bestimmt die Review-Qualität basierend auf Antwort-Korrektheit, Zeit und Hinweisen

    Args:
        is_correct: Ob die Antwort korrekt war
        response_time: Antwortzeit in Sekunden
        hints_used: Anzahl verwendeter Hinweise

    Returns:
        ReviewQuality enum
    """
    if not is_correct:
        return ReviewQuality.INCORRECT

    # Basierend auf Zeit und Hinweisen bewerten
    if hints_used == 0 and response_time <= 10:
        return ReviewQuality.PERFECT
    elif hints_used == 0 and response_time <= 30:
        return ReviewQuality.EASY
    elif hints_used <= 1 and response_time <= 60:
        return ReviewQuality.GOOD
    else:
        return ReviewQuality.HARD
