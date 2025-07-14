#!/bin/bash

# Container-Start-Skript für den Lernagenten

echo "🤖 Universeller Lernagent - Container Start"
echo "=========================================="

# Verzeichnisse prüfen und erstellen
echo "📁 Überprüfe Datenverzeichnisse..."
mkdir -p /app/models /app/scenarios /app/data /app/data/spaced_repetition

# Beispiel-Szenarien erstellen falls nicht vorhanden
echo "📚 Überprüfe Szenarien..."
if [ ! -f "/app/scenarios/latein_anfaenger.json" ]; then
    echo "Erstelle Beispiel-Szenarien..."
    python -c "
from src.learning_agent.scenario_manager import ScenarioManager
sm = ScenarioManager()
sm.create_sample_scenarios()
print('Beispiel-Szenarien erstellt')
"
fi

# Verfügbare Szenarien anzeigen
echo "📋 Verfügbare Szenarien:"
python -c "
from src.learning_agent.scenario_manager import ScenarioManager
sm = ScenarioManager()
scenarios = sm.list_scenarios()
for scenario in scenarios:
    print(f'  - {scenario}')
"

echo ""
echo "🚀 Starte Learning Agent API..."
echo "   Web-Interface: http://localhost:8001"
echo "   API Docs: http://localhost:8001/docs" 
echo "   Gesundheitscheck: http://localhost:8001/api/health"
echo ""

# API Server starten
exec python -m src.learning_agent.api
