import os
import csv
from datetime import datetime
from attendance.excel_generator import generate_excel
ATTENDANCE_DIR = "data/attendance"
ATTENDANCE_FILE = os.path.join(ATTENDANCE_DIR,"attendance.csv",)


class AttendanceManager:

    def __init__(self):

        os.makedirs(
            ATTENDANCE_DIR,
            exist_ok=True,
        )

        self.marked_students = set()

        self.load_existing_attendance()

    def load_existing_attendance(self):

        if not os.path.exists(ATTENDANCE_FILE):
            return

        with open(ATTENDANCE_FILE,"r",newline="",encoding="utf-8",) as file:

            reader = csv.DictReader(file)

            today = datetime.now().strftime("%Y-%m-%d")

            for row in reader:

                if (
                    row.get("Date") == today
                    and row.get("Status") == "Present"
                ):
                    self.marked_students.add(
                        row.get("Name")
                    )

    def mark_attendance(self, name):

        if name == "UNKNOWN":
            return False

        if name in self.marked_students:
            return False

        now = datetime.now()

        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")

        file_exists = os.path.exists(
            ATTENDANCE_FILE
        )

        with open(ATTENDANCE_FILE,"a",newline="",encoding="utf-8",) as file:

            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(
                    [
                        "Name",
                        "Date",
                        "Time",
                        "Status",
                    ]
                )

            writer.writerow(
                [
                    name,
                    date,
                    time,
                    "Present",
                ]
            )

        self.marked_students.add(name)
        generate_excel()

        print(
            f"Attendance marked: {name} "
            f"at {time}"
        )

        return True