import sys
import os
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.attendance.attendance_manager import AttendanceManager


def main():
    attendance = AttendanceManager()

    print("\nAttendance Test")
    print("----------------")

    # First attempt
    result1 = attendance.mark_attendance("balaji")

    # Duplicate attempt
    result2 = attendance.mark_attendance("balaji")

    # Unknown person
    result3 = attendance.mark_attendance("UNKNOWN")

    print(f"\nFirst attempt: {'Marked' if result1 else 'Not marked'}")
    print(f"Duplicate attempt: {'Marked' if result2 else 'Not marked'}")
    print(f"Unknown person: {'Marked' if result3 else 'Not marked'}")

    if result1 and not result2 and not result3:
        print("\nPASS: Attendance logic works correctly.")
    else:
        print("\nFAIL: Attendance logic has a problem.")


if __name__ == "__main__":
    main()