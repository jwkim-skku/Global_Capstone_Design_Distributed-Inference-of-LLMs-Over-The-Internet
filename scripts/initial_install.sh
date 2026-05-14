#!/bin/bash
# initial_install.sh - 가상환경 + 의존성 설치

set -e

# 프로젝트 루트로 이동 (스크립트 위치 기준)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

echo "=========================================="
echo "의존성 설치 스크립트"
echo "=========================================="
echo "프로젝트 루트: $PROJECT_ROOT"
echo "=========================================="

# requirements.txt 확인
if [ ! -f "requirements.txt" ]; then
    echo "오류: requirements.txt를 찾을 수 없습니다."
    echo "프로젝트 루트에서 실행해주세요: $PROJECT_ROOT"
    exit 1
fi

# Python 가상환경 확인 및 생성
if [ ! -d "venv" ]; then
    echo "가상환경 생성 중..."
    python3 -m venv venv
else
    echo "가상환경이 이미 존재합니다."
fi

# 가상환경 활성화
echo "가상환경 활성화 중..."
source venv/bin/activate

# 의존성 설치
echo "의존성 설치 중..."
pip install --upgrade pip
# hivemind 소스 빌드 시 PEP517 격리 환경에서 pkg_resources 누락 오류 방지
pip install --upgrade "setuptools>=69" wheel
pip install -r requirements.txt

# hivemind 소스 빌드 시 proto_compile 단계에서 grpc_tools 필요
pip install "grpcio>=1.56" "grpcio-tools>=1.56"

# hivemind는 소스에서 직접 설치 (플랫폼별 바이너리 문제 방지)
# 버전 1.1.11로 고정
# --no-build-isolation: 위에서 설치한 setuptools( pkg_resources 포함 )로 setup.py 실행
INSTALLED_VERSION=$(pip show hivemind 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
if [ "$INSTALLED_VERSION" = "1.1.11" ]; then
    echo "hivemind 1.1.11이 이미 설치되어 있습니다. 설치 단계를 건너뜁니다."
else
    echo "hivemind 1.1.11 설치 중 (소스에서 빌드)..."
    pip install --force-reinstall --no-binary=hivemind --no-build-isolation hivemind==1.1.11
fi

echo "=========================================="
echo "설치 완료!"
echo "=========================================="
echo "가상환경 활성화: source venv/bin/activate"