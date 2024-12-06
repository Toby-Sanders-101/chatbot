import os
import json
from email.parser import Parser

# Directory of the extracted Enron dataset
enron_data_dir = "enron_mail"
output_json_file = "enron_emails.json"

def parse_email(file_path):
    """
    Parses an individual email file into a dictionary.
    """
    with open(file_path, "r", encoding="latin1") as f:
        raw_email = f.read()
    
    # Parse the email headers and body
    email = Parser().parsestr(raw_email)
    parsed_email = {
        "from": email.get("From"),
        "to": email.get("To"),
        "subject": email.get("Subject"),
        "date": email.get("Date"),
        "body": email.get_payload()  # Handles multi-part emails as raw text
    }
    return parsed_email

def parse_directory(directory):
    """
    Recursively parses all email files in the dataset directory.
    """
    email_list = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                email_data = parse_email(file_path)
                email_list.append(email_data)
            except Exception as e:
                print(f"Failed to parse {file_path}: {e}")
    return email_list

def main():
    if not os.path.exists(enron_data_dir):
        print(f"Error: Directory '{enron_data_dir}' does not exist. Please extract the dataset first.")
        return

    print("Parsing emails...")
    all_emails = parse_directory(enron_data_dir)

    print(f"Saving parsed emails to {output_json_file}...")
    with open(output_json_file, "w", encoding="utf-8") as json_file:
        json.dump(all_emails, json_file, indent=4)

    print(f"Parsing complete. JSON file saved as {output_json_file}.")

if __name__ == "__main__":
    main()
