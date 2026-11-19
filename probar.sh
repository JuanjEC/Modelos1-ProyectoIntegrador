#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
image="modelos-api:3.0"
container="modelos-prueba-$$"
run_dir="outputs/ensayo-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run_dir" models

echo "== Construccion y pruebas"
docker build -t "$image" .
docker run --rm "$image" python -m pytest -q

cleanup() {
  docker rm -f "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d --name "$container" \
  -e MODEL_VERSION=v2 -e PREDICTION_LOG=/app/registro/solicitudes.jsonl \
  -e GIT_COMMIT="$(git rev-parse HEAD)" \
  -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models" \
  -v "$PWD/$run_dir:/app/registro" "$image" >/dev/null

ready=false
for attempt in $(seq 1 60); do
  if docker exec "$container" python -c "from urllib.request import urlopen; print(urlopen('http://localhost:8000/health').read().decode())" 2>/dev/null; then
    ready=true
    break
  fi
  sleep 1
done
if [ "$ready" != true ]; then
  docker logs "$container"
  exit 1
fi

echo "== Dos predicciones validas y una sin Mileage"
docker exec "$container" python fase-3/cliente.py
docker exec "$container" python fase-3/cliente.py
if docker exec "$container" python fase-3/cliente.py --invalid; then
  echo "ERROR: se acepto la entrada invalida"
  exit 1
fi

echo "== Monitoreo (debe mostrar 3 solicitudes y 33,3 % de fallos)"
docker exec "$container" python fase-4/monitoreo.py --log /app/registro/solicitudes.jsonl
docker exec "$container" python -c "import json; from pathlib import Path; rows=[json.loads(line) for line in Path('/app/registro/solicitudes.jsonl').read_text().splitlines()]; assert [row['estado_http'] for row in rows]==[200,200,422]; assert all(row['version']=='v2' for row in rows[:2])"

echo "== Entrenamiento HTTP y verificacion de artefactos persistentes"
docker exec "$container" python -c "import json; from pathlib import Path; from urllib.request import Request,urlopen; response=urlopen(Request('http://localhost:8000/train',data=b'{}',headers={'Content-Type':'application/json'}),timeout=1800); result=json.load(response); version=result['version']; print('HTTP',response.status); print(json.dumps(result,ensure_ascii=False)); assert all((Path('/app/models')/version/name).is_file() for name in ['modelo.joblib','metrics.json','metadata.json'])"

echo "== Fin. Registro de esta ejecucion: $run_dir/solicitudes.jsonl"
