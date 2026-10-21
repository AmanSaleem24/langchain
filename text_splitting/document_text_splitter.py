from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader('LostLink_Review_Paper.pdf')

docs = loader.load()

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 100, 
    chunk_overlap = 10,
)

text = """

import json
import logging
from datetime import datetime
from pathlib import Path
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info("Executing %s", func.__name__)
        result = func(*args, **kwargs)
        return result
    return wrapper


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"My name is {self.name} and I am {self.age} years old."


class Student(Person):
    def __init__(self, name, age, student_id, courses=None):
        super().__init__(name, age)
        self.student_id = student_id
        self.courses = courses or []
        self.grades = {}

    def enroll(self, course):
        if course not in self.courses:
            self.courses.append(course)
            logger.info("%s enrolled in %s", self.name, course)

    def add_grade(self, course, grade):
        if course not in self.courses:
            raise ValueError("Student is not enrolled in this course.")
        if not 0 <= grade <= 100:
            raise ValueError("Grade must be between 0 and 100.")
        self.grades[course] = grade

    def calculate_average(self):
        if not self.grades:
            return 0.0
        return sum(self.grades.values()) / len(self.grades)

    def to_dict(self):
        return {
            "name": self.name,
            "age": self.age,
            "student_id": self.student_id,
            "courses": self.courses,
            "grades": self.grades,
            "average": self.calculate_average(),
        }


class StudentManager:
    def __init__(self):
        self.students = {}

    def add_student(self, student):
        if student.student_id in self.students:
            raise ValueError("Student ID already exists.")
        self.students[student.student_id] = student

    def get_student(self, student_id):
        return self.students.get(student_id)

    def remove_student(self, student_id):
        if student_id in self.students:
            return self.students.pop(student_id)
        return None

    @log_execution
    def display_students(self):
        if not self.students:
            print("No students registered.")
            return

        for student in self.students.values():
            print(student.introduce())
            print(f"ID: {student.student_id}")
            print(f"Courses: {', '.join(student.courses)}")
            print(f"Average grade: {student.calculate_average():.2f}")
            print("-" * 40)

    def save_to_file(self, filename):
        data = [
            student.to_dict()
            for student in self.students.values()
        ]

        path = Path(filename)
        try:
            path.write_text(json.dumps(data, indent=4), encoding="utf-8")
            logger.info("Data saved to %s", path)
        except OSError as error:
            logger.error("Unable to save data: %s", error)
            raise

    def load_from_file(self, filename):
        path = Path(filename)

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data:
                student = Student(
                    item["name"],
                    item["age"],
                    item["student_id"],
                    item["courses"],
                )
                student.grades = item["grades"]
                self.students[student.student_id] = student

        except FileNotFoundError:
            logger.warning("Data file does not exist.")
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            logger.error("Invalid student data: %s", error)
            raise


def create_sample_data(manager):
    students = [
        Student("Alice", 20, "S101", ["Python", "Mathematics"]),
        Student("Bob", 21, "S102", ["Java", "Database Systems"]),
        Student("Charlie", 19, "S103", ["Python", "Machine Learning"]),
    ]

    for student in students:
        manager.add_student(student)

    manager.get_student("S101").add_grade("Python", 92)
    manager.get_student("S101").add_grade("Mathematics", 85)
    manager.get_student("S102").add_grade("Java", 78)
    manager.get_student("S102").add_grade("Database Systems", 88)
    manager.get_student("S103").add_grade("Python", 95)
    manager.get_student("S103").add_grade("Machine Learning", 91)


def main():
    manager = StudentManager()
    create_sample_data(manager)

    print("STUDENT MANAGEMENT SYSTEM")
    print("=" * 40)

    manager.display_students()

    try:
        student = manager.get_student("S101")
        if student:
            student.enroll("Artificial Intelligence")
            student.add_grade("Artificial Intelligence", 94)

        manager.save_to_file("students.json")
        print("Student records saved successfully.")

    except (ValueError, OSError) as error:
        print(f"An error occurred: {error}")

    print(f"Report generated at {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()

"""


chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---\n")
    print(chunk)

