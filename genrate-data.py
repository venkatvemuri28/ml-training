import csv

# Sample data for the CSV file
data = [
    {
        "ticket_text": "User is unable to login with credentials and sees an error message 'invalid credentials'.",
        "ticket_summary": "Login failure due to credentials error."
    },
    {
        "ticket_text": "Application crashes when the user opens the dashboard after the latest update. The issue seems to occur consistently.",
        "ticket_summary": "Crash on dashboard load after update."
    },
    {
        "ticket_text": "The system experiences slow performance and occasional freezing during peak hours. Users report delays in processing requests.",
        "ticket_summary": "Performance degradation and intermittent freezing during peak hours."
    },
    {
        "ticket_text": "Printer not responding despite being connected to the network. The error indicates a connection timeout.",
        "ticket_summary": "Network printer connection timeout issue."
    },
    {
        "ticket_text": "Website is not loading at all, and users are encountering a 504 gateway timeout error. It appears to affect all pages.",
        "ticket_summary": "Website downtime due to 504 gateway timeout error."
    }
]

# Specify the file name and the field names that correspond to the ticket details and summaries.
csv_file = "ticket_data.csv"
fieldnames = ["ticket_text", "ticket_summary"]

# Write the sample data to the CSV file.
with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

print(f"Sample CSV file '{csv_file}' has been created successfully.")
