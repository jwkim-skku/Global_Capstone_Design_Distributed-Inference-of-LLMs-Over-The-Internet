#!/usr/bin/env python3
"""
특정 stage 서버를 종료하는 헬퍼 스크립트

사용법:
    python scripts/kill_stage.py 1  # stage1 종료
    python scripts/kill_stage.py 2  # stage2 종료
    python scripts/kill_stage.py 3  # stage3 종료
"""

import argparse
import subprocess
import sys


def kill_stage(stage_num: int):
    """특정 stage의 프로세스를 찾아서 종료"""
    try:
        # ps 명령으로 stage 프로세스 찾기
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            check=True
        )
        
        lines = result.stdout.split("\n")
        stage_processes = []
        
        for line in lines:
            if f"--stage {stage_num}" in line and "python" in line:
                parts = line.split()
                if parts:
                    pid = parts[1]
                    stage_processes.append((pid, line))
        
        if not stage_processes:
            print(f"Stage {stage_num} 프로세스를 찾을 수 없습니다.")
            return False
        
        print(f"Stage {stage_num} 프로세스 {len(stage_processes)}개 발견:")
        for pid, cmd in stage_processes:
            print(f"  PID {pid}: {cmd[:80]}...")
        
        # 사용자 확인
        response = input(f"\nStage {stage_num} 프로세스를 종료하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("취소되었습니다.")
            return False
        
        # 프로세스 종료
        for pid, _ in stage_processes:
            try:
                subprocess.run(["kill", "-TERM", pid], check=True)
                print(f"PID {pid} 종료 신호 전송 완료")
            except subprocess.CalledProcessError as e:
                print(f"PID {pid} 종료 실패: {e}")
        
        print(f"\nStage {stage_num} 종료 완료. 클라이언트가 복구를 시도할 것입니다.")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"오류 발생: {e}")
        return False
    except KeyboardInterrupt:
        print("\n취소되었습니다.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Kill a specific stage server for fault tolerance testing")
    parser.add_argument("stage", type=int, choices=[1, 2, 3], help="Stage number to kill (1, 2, or 3)")
    args = parser.parse_args()
    
    kill_stage(args.stage)


if __name__ == "__main__":
    main()

