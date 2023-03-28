import sys
sys.stdin = open("test.txt")

for line in sys.stdin:
    print(line)

user_input = input("Please enter some text: ")
print("You entered:", user_input) 
