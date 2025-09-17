print("Grading Program")
print("Enter each subject with its marks. Type 'done' when you are finished.\n")

total_marks = 0
subject_count = 0
records = []

while True:
    subject = input("Subject name (or type 'done' to finish): ").strip()
    
    if subject.lower() == "done":
        break
    
    try:
        marks = float(input(f"Marks in {subject}: "))
    except ValueError:
        print("Please enter a valid number for marks.\n")
        continue
    
    # Assign grade
    if marks >= 90:
        grade = "A"
    elif marks >= 80:
        grade = "B"
    elif marks >= 70:
        grade = "C"
    elif marks >= 60:
        grade = "D"
    else:
        grade = "F"
    
    print(f"{subject}: {marks} → Grade {grade}\n")
    
    records.append((subject, marks, grade))
    total_marks += marks
    subject_count += 1

# Final summary
if subject_count > 0:
    average = total_marks / subject_count
    
    if average >= 90:
        final_grade = "A"
    elif average >= 80:
        final_grade = "B"
    elif average >= 70:
        final_grade = "C"
    elif average >= 60:
        final_grade = "D"
    else:
        final_grade = "F"
    
    print("\n--- Final Report ---")
    for sub, marks, grade in records:
        print(f"{sub:20} {marks:5}   Grade: {grade}")
    
    print("\nTotal Subjects :", subject_count)
    print("Total Marks    :", total_marks)
    print("Average Score  :", round(average, 2))
    print("Final Grade    :", final_grade)
else:
    print("No subjects were entered.")
