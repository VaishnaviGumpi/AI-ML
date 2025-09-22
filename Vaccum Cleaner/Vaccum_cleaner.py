# Simple Vacuum Cleaner Program

print("Vacuum Cleaner Models:")
print("1. Circle  - Good at moving around furniture")
print("2. Square  - Cleans along walls nicely")
print("3. Star    - Can reach sharp corners")
print("4. Oval    - Smooth movement in open space")

choice = input("\nPick a model (1/2/3/4): ")

if choice == "1":
    model = "Circle"
elif choice == "2":
    model = "Square"
elif choice == "3":
    model = "Star"
elif choice == "4":
    model = "Oval"
else:
    model = "Circle"
    print("Invalid option! Defaulting to Circle.")

print("\nYou chose:", model, "Vacuum Cleaner")

print("\nCommands: start, stop, left, right, dock")
print("Type 'exit' to finish.\n")

while True:
    cmd = input("Command: ")

    if cmd == "exit":
        print("Cleaning done. Bye!")
        break
    elif cmd == "start":
        print(model, "vacuum started.")
    elif cmd == "stop":
        print(model, "vacuum stopped.")
    elif cmd == "left":
        print(model, "vacuum turned left.")
    elif cmd == "right":
        print(model, "vacuum turned right.")
    elif cmd == "dock":
        print(model, "vacuum returned to dock.")
    else:
        print("Not a valid command, try again.")
