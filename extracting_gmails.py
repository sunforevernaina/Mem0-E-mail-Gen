import os
import base64
import pandas as pd
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email import message_from_bytes
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from html import unescape
import json
import re

# Gmail API Scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def gmail_authenticate():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def extract_plain_text_from_payload(payload):
    """Extract and decode the plain text content from the email payload."""
    if payload.get('mimeType') == 'text/plain':
        body_data = payload.get('body', {}).get('data')
        if body_data:
            return decode_base64_text(body_data)

    elif payload.get('mimeType', '').startswith('multipart'):
        for part in payload.get('parts', []):
            text = extract_plain_text_from_payload(part)
            if text:
                return text
    return ""

def decode_base64_text(encoded_text):
    """Decode Base64-encoded email text safely."""
    try:
        decoded_bytes = base64.urlsafe_b64decode(encoded_text.encode('UTF-8'))
        decoded_text = decoded_bytes.decode('utf-8', errors='ignore')
        return unescape(decoded_text.strip())
    except Exception as e:
        return f"[Decoding error: {str(e)}]"
    
def extract_first_email(text: str) -> str:
    text = text.replace('\r\n', '\n')

    split_markers = [
        r'\nFrom: ',           # e.g. "From: Sunaina Saxena <...>"
        r'\nOn .* wrote:',     # e.g. "On Fri, 13 Jun 2025 at 14:46, ..."
    ]

    split_pattern = re.compile('|'.join(split_markers), re.IGNORECASE)

    # Split at the first occurrence of any marker
    split = split_pattern.split(text, maxsplit=1)
    first_message = split[0].strip()

    # Clean excessive newlines and join lines for better readability
    cleaned = re.sub(r'\n+', ' ', first_message)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned

def get_snippets_with_person(service, target_email, user_id='me', max_results=100, output_file='conversation_snippets.json'):
    messages = service.users().messages().list(userId=user_id, maxResults=max_results).execute().get('messages', [])
    conversation_snippets = []

    for msg in messages:
        try:
            msg_data = service.users().messages().get(userId=user_id, id=msg['id'], format='full').execute()
            payload = msg_data.get("payload", {})
            headers = payload.get("headers", [])

            email_info = {
                'body': '',
                'from': '',
                'to': '',
                'subject': '',
                'date': ''
            }

            for header in headers:
                name = header['name'].lower()
                value = header['value']
                if name == 'from':
                    email_info['from'] = value
                elif name == 'to':
                    email_info['to'] = value
                elif name == 'subject':
                    email_info['subject'] = value
                elif name == 'date':
                    email_info['date'] = value

            

            # Filter by target email
            if target_email.lower() not in email_info['from'].lower() and \
               target_email.lower() not in email_info['to'].lower():
                continue
             # Extract and decode full body text
            email_info['body'] = extract_first_email(extract_plain_text_from_payload(payload))
            conversation_snippets.append(email_info)

        except HttpError as error:
            print(f"An error occurred: {error}")

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversation_snippets, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(conversation_snippets)} messages involving '{target_email}' to '{output_file}'.")

    return conversation_snippets


if __name__ == "__main__":
    gmail_service = gmail_authenticate()
    target_email = input("Enter the email address to filter: ")
    get_snippets_with_person(gmail_service, target_email)
    print("All the emails from " + target_email + " are fetched")
