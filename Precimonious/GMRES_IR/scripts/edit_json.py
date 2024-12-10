import json
import argparse

def update_json_file(filename, n, condition_number):
    try:
        # Load existing data from JSON file
        with open(filename, 'r') as file:
            data = json.load(file)
        
        # Update the fields
        data['n'] = n
        data['condition_number'] = condition_number
        
        # Save the updated data back to the JSON file
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
        
        print(f"Updated '{filename}' with n={n} and condition_number={condition_number}.")
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON file.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Update fields in settings.json.")
    parser.add_argument("n", type=int, help="The value for n")
    parser.add_argument("condition_number", type=float, help="The value for condition_number")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Update the JSON file
    update_json_file("settings.json", args.n, args.condition_number)
