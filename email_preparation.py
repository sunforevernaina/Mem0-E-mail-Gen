
# Email & Calendar Agent Framework using LangGraph
import json
import streamlit as st
from typing import Dict, List, Any, TypedDict, Annotated
from dataclasses import dataclass
from enum import Enum
import google.generativeai as genai
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage
import datetime

import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

# Configure Gemini
apikey = os.getenv('GEMINI-API-KEY')
genai.configure(api_key=apikey)

class AgentState(TypedDict):
    """State shared across all agents"""
    messages: Annotated[List[BaseMessage], add_messages]
    memory_data: Dict[str, Any]
    team_members: List[Dict[str, str]]
    email_content: str
    meeting_required: bool
    meeting_details: Dict[str, Any]
    email_recipients: List[str]
    calendar_invite_sent: bool
    final_output: Dict[str, Any]
    current_step: str
    errors: List[str]
    sender_email: str
    llm_prompt: str
    selected_recipients: List[str]  # Add this line

@dataclass
class TeamMember:
    name: str
    email: str
    role: str
    project: str

class EmailTemplate:
    """Email templates for different scenarios"""

    @staticmethod
    def project_update_template(context: Dict) -> str:
        return f"""
Subject: Project Update - {context.get('project_name', 'DIA MVP')}

Hi Team,

I hope this email finds you well. I wanted to share some updates regarding our project:

{context.get('main_content', '')}

{context.get('action_items', '')}

{context.get('meeting_info', '')}

Best regards,
{context.get('sender_name', 'Anshika')}
        """

    @staticmethod
    def meeting_invitation_template(context: Dict) -> str:
        return f"""
Subject: Meeting Invitation - {context.get('meeting_subject', 'Project Discussion')}

Hi Team,

I'd like to schedule a meeting to discuss our project progress and next steps.

Meeting Details:
- Date: {context.get('date', '')}
- Time: {context.get('time', '')}
- Duration: {context.get('duration', '1 hour')}
- Platform: {context.get('platform', 'Google Meet')}

Agenda:
{context.get('agenda', '')}

Please confirm your availability.

Best regards,
{context.get('sender_name', 'Anshika')}
        """

class MemoryAnalyzer:
    """Analyzes Neo4j memory to extract actionable insights"""

    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')

    def analyze_memory_context(self, memory_data: Dict) -> Dict[str, Any]:
        """Analyze memory data to extract context for email drafting"""

        # Extract key information
        memories = memory_data.get('results', [])
        relations = memory_data.get('relations', {})

        context = {
            'action_items': [],
            'meetings_required': False,
            'project_updates': [],
            'key_entities': set(),
            'relationships': []
        }

        # Process memories
        for memory in memories:
            memory_text = memory.get('memory', '')
            if 'calendar' in memory_text.lower() or 'meeting' in memory_text.lower():
                context['meetings_required'] = True
                context['action_items'].append(memory_text)
            else:
                context['project_updates'].append(memory_text)

        # Process relationships
        added_entities = relations.get('added_entities', [])
        for entity_group in added_entities:
            for entity in entity_group:
                context['key_entities'].add(entity.get('source', ''))
                context['key_entities'].add(entity.get('target', ''))
                context['relationships'].append(entity)

        return context

class EmailDraftingAgent:
    """Agent responsible for drafting emails based on memory context"""

    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.memory_analyzer = MemoryAnalyzer()

    def draft_email(self, state: AgentState) -> AgentState:
        """Draft email based on memory context and team information"""
        try:
            # Analyze memory context
            context = self.memory_analyzer.analyze_memory_context(state['memory_data'])

            # Get sender info
            sender = next((m for m in state['team_members'] if m['email'] == state.get('sender_email')), None)
            sender_name = sender['name'] if sender else "Anshika"

            # Prepare prompt for Gemini
            prompt = f"""
            Based on the following context from memory data, draft a professional email:

            Sender: {sender_name}
            Recipients: {[member['name'] for member in state['team_members'] if member['email'] in state['selected_recipients']]}

            Memory Context:
            - Project Updates: {context['project_updates']}
            - Action Items: {context['action_items']}
            - Key Relationships: {context['relationships']}

            Requirements:
            1. Professional tone
            2. Clear action items
            3. Relevant project context
            4. Appropriate greetings and closings
            5. Signature with sender's name

            Draft a concise and actionable email:
            """

            # Store the prompt for display
            state['llm_prompt'] = prompt

            response = self.model.generate_content(prompt)
            email_content = response.text

            # Check if meeting is required
            meeting_required = any('meeting' in item.lower() or 'calendar' in item.lower()
                                 for item in context['action_items'])

            state['email_content'] = email_content
            state['meeting_required'] = meeting_required
            state['current_step'] = 'email_drafted'

        except Exception as e:
            state['errors'].append(f"Email drafting error: {str(e)}")
            state['current_step'] = 'error'

        return state

class MeetingSchedulerAgent:
    """Agent responsible for handling meeting scheduling logic"""

    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')

    def extract_meeting_details(self, state: AgentState) -> AgentState:
        """Extract meeting details from memory and email content"""

        try:
            if not state['meeting_required']:
                state['current_step'] = 'meeting_not_required'
                return state

            # Extract meeting details from memory
            memory_relations = state['memory_data'].get('relations', {})
            meeting_time = None

            for entity_group in memory_relations.get('added_entities', []):
                for entity in entity_group:
                    if 'meeting_time' in entity.get('relationship', ''):
                        meeting_time = entity.get('target', '')

            # Use Gemini to structure meeting details
            prompt = f"""
            Extract meeting details from the following context:

            Memory Data: {state['memory_data']}
            Email Content: {state['email_content']}
            Meeting Time Found: {meeting_time}

            Provide meeting details in JSON format:
            {{
                "subject": "meeting subject",
                "date": "proposed date",
                "time": "proposed time",
                "duration": "duration",
                "agenda": ["agenda item 1", "agenda item 2"]
            }}
            """

            response = self.model.generate_content(prompt)
            meeting_details = json.loads(response.text.strip('```json\n').strip('```'))

            state['meeting_details'] = meeting_details
            state['current_step'] = 'meeting_details_extracted'

        except Exception as e:
            state['errors'].append(f"Meeting scheduling error: {str(e)}")
            state['current_step'] = 'error'

        return state

class CalendarIntegrationTool(BaseTool):
    """Tool for calendar integration (mock implementation)"""

    name: str = "calendar_integration"
    description: str = "Sends calendar invites to team members"

    def _run(self, meeting_details: Dict, recipients: List[str]) -> str:
        """Mock calendar integration - replace with actual Google Calendar API"""

        # This would integrate with Google Calendar API
        # For now, returning mock response
        invite_details = {
            "meeting_id": f"meeting_{datetime.datetime.now().timestamp()}",
            "subject": meeting_details.get("subject", "Team Meeting"),
            "attendees": recipients,
            "date": meeting_details.get("date", ""),
            "time": meeting_details.get("time", ""),
            "status": "sent"
        }

        return f"Calendar invite sent successfully: {json.dumps(invite_details)}"

class EmailSenderTool(BaseTool):
    """Tool for sending emails (mock implementation)"""

    name: str = "email_sender"
    description: str = "Sends emails to team members"

    def _run(self, email_content: str, recipients: List[str], subject: str = None) -> str:
        """Mock email sending - replace with actual email service"""

        # This would integrate with Gmail API or SMTP
        email_details = {
            "email_id": f"email_{datetime.datetime.now().timestamp()}",
            "recipients": recipients,
            "subject": subject or "Project Update",
            "content_preview": email_content[:100] + "...",
            "status": "sent"
        }

        return f"Email sent successfully: {json.dumps(email_details)}"

class EmailCalendarOrchestrator:
    """Main orchestrator class that coordinates all agents"""

    def __init__(self):
        self.email_agent = EmailDraftingAgent()
        self.meeting_agent = MeetingSchedulerAgent()

        # Initialize tools
        self.calendar_tool = CalendarIntegrationTool()
        self.email_tool = EmailSenderTool()
        self.available_tools = [self.calendar_tool, self.email_tool]

        # Build the graph
        self.workflow = self._build_workflow()

    def _execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        for tool in self.available_tools:
            if tool.name == tool_name:
                return tool._run(**kwargs)
        raise ValueError(f"Tool {tool_name} not found")

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_memory", self._analyze_memory_node)
        workflow.add_node("draft_email", self._draft_email_node)
        workflow.add_node("extract_meeting", self._extract_meeting_node)
        workflow.add_node("send_email", self._send_email_node)
        workflow.add_node("send_calendar", self._send_calendar_node)
        workflow.add_node("finalize", self._finalize_node)

        # Set entry point
        workflow.set_entry_point("analyze_memory")

        # Add edges
        workflow.add_edge("analyze_memory", "draft_email")
        workflow.add_edge("draft_email", "extract_meeting")

        # Conditional edges
        workflow.add_conditional_edges(
            "extract_meeting",
            self._should_schedule_meeting,
            {
                "schedule_meeting": "send_calendar",
                "skip_meeting": "send_email"
            }
        )

        workflow.add_edge("send_calendar", "send_email")
        workflow.add_edge("send_email", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _analyze_memory_node(self, state: AgentState) -> AgentState:
        """Node for memory analysis"""
        state['current_step'] = 'analyzing_memory'
        return state

    def _draft_email_node(self, state: AgentState) -> AgentState:
        """Node for email drafting"""
        return self.email_agent.draft_email(state)

    def _extract_meeting_node(self, state: AgentState) -> AgentState:
        """Node for meeting extraction"""
        return self.meeting_agent.extract_meeting_details(state)

    def _send_email_node(self, state: AgentState) -> AgentState:
        """Node for sending email"""
        try:
            recipients = state['selected_recipients']  # Use the selected_recipients directly
            result = self._execute_tool(
                "email_sender",
                email_content=state['email_content'],
                recipients=recipients,
                subject="Project Update"
            )
            state['current_step'] = 'email_sent'
            if 'final_output' not in state:
                state['final_output'] = {}
            state['final_output']['email_result'] = result
        except Exception as e:
            state['errors'].append(f"Email sending error: {str(e)}")
        return state

    def _send_calendar_node(self, state: AgentState) -> AgentState:
        """Node for sending calendar invite"""
        try:
            if state['meeting_details']:
                recipients = state['selected_recipients']  # Use the selected_recipients directly
                result = self._execute_tool(
                    "calendar_integration",
                    meeting_details=state['meeting_details'],
                    recipients=recipients
                )
                state['calendar_invite_sent'] = True
                if 'final_output' not in state:
                    state['final_output'] = {}
                state['final_output']['calendar_result'] = result
        except Exception as e:
            state['errors'].append(f"Calendar error: {str(e)}")
        return state

    def _finalize_node(self, state: AgentState) -> AgentState:
        """Final processing node"""
        state['current_step'] = 'completed'
        state['final_output']['summary'] = {
            'email_sent': 'email_result' in state['final_output'],
            'calendar_sent': state['calendar_invite_sent'],
            'recipients_count': len(state['selected_recipients']),
            'meeting_required': state['meeting_required']
        }
        return state

    def _should_schedule_meeting(self, state: AgentState) -> str:
        """Conditional logic for meeting scheduling"""
        if state['meeting_required'] and state.get('meeting_details'):
            return "schedule_meeting"
        return "skip_meeting"

    def run_workflow(self, memory_data: Dict, team_members: List[Dict], sender_email: str, selected_recipients: List[str]) -> Dict:
        """Execute the complete workflow"""
        initial_state = AgentState(
            messages=[],
            memory_data=memory_data,
            team_members=team_members,
            email_content="",
            meeting_required=False,
            meeting_details={},
            email_recipients=selected_recipients,  # Changed this line
            calendar_invite_sent=False,
            final_output={},
            current_step="starting",
            errors=[],
            sender_email=sender_email,
            llm_prompt="",
            selected_recipients=selected_recipients  # Add this line
        )

        # Run the workflow
        result = self.workflow.invoke(initial_state)
        return result

# Sample data for demonstration
SAMPLE_MEMORY_DATA = {
    "results": [
        {
            "id": "2391a7a0-21de-400c-b3fc-f9a34d3a02b1",
            "memory": "Will look out for calendar invite and deck",
            "event": "UPDATE",
            "previous_memory": "Will receive calendar invite with brief deck attached"
        },
        {
            "id": "0519bc0d-75cd-4453-bf31-e1b18b2fdf5b",
            "memory": "Looking forward to learning more about DIA and contributing to shaping the MVP",
            "event": "ADD"
        }
    ],
    "relations": {
        "deleted_entities": [],
        "added_entities": [
            [{"source": "anshika", "relationship": "sent_email_to", "target": "sunaina"}],
            [{"source": "anshika", "relationship": "looking_for", "target": "calendar_invite"}],
            [{"source": "anshika", "relationship": "looking_for", "target": "deck"}],
            [{"source": "anshika", "relationship": "wants_to_learn_about", "target": "dia"}],
            [{"source": "anshika", "relationship": "wants_to_contribute_to", "target": "mvp"}],
            [{"source": "anshika", "relationship": "meeting_time", "target": "tuesday_at_10am_pt"}]
        ]
    }
}

def create_sample_team_csv():
    """Create a sample team CSV file for demonstration"""
    sample_team_data = {
        'name': ['Anshika Sharma', 'Sunaina Patel', 'Rahul Kumar', 'Priya Singh', 'Amit Gupta', 'Neha Agarwal'],
        'email': ['anshika.sharma@company.com', 'sunaina.patel@company.com', 'rahul.kumar@company.com',
                 'priya.singh@company.com', 'amit.gupta@company.com', 'neha.agarwal@company.com'],
        'role': ['Product Manager', 'Tech Lead', 'Senior Developer', 'UI/UX Designer', 'Data Scientist', 'QA Engineer'],
        'department': ['Product', 'Engineering', 'Engineering', 'Design', 'Data', 'Quality'],
        'project': ['DIA MVP', 'DIA MVP', 'DIA MVP', 'DIA MVP', 'DIA MVP', 'DIA MVP'],
        'location': ['Bengaluru', 'Mumbai', 'Bengaluru', 'Delhi', 'Bengaluru', 'Mumbai'],
        'phone': ['+91-9876543210', '+91-9876543211', '+91-9876543212', '+91-9876543213', '+91-9876543214', '+91-9876543215']
    }

    df = pd.DataFrame(sample_team_data)
    return df

def load_team_from_csv(uploaded_file=None):
    """Load team members from CSV file"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            # Use sample data if no file uploaded
            df = create_sample_team_csv()

        # Convert to list of dictionaries
        team_members = []
        for _, row in df.iterrows():
            member = {
                'name': str(row.get('name', '')),
                'email': str(row.get('email', '')),
                'role': str(row.get('role', '')),
                'department': str(row.get('department', '')),
                'project': str(row.get('project', 'DIA MVP')),
                'location': str(row.get('location', '')),
                'phone': str(row.get('phone', ''))
            }
            if member['name'] and member['email']:
                team_members.append(member)

        return team_members, None

    except Exception as e:
        return [], f"Error loading team data: {str(e)}"

# Streamlit UI
def create_streamlit_ui():
    """Create Streamlit interface for the email/calendar system"""

    st.title("🤖 AI Email & Calendar Assistant")
    st.markdown("*Powered by LangGraph & Gemini*")

    # Initialize orchestrator
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = EmailCalendarOrchestrator()

    # Initialize session state for data
    if 'memory_data' not in st.session_state:
        st.session_state.memory_data = SAMPLE_MEMORY_DATA

    if 'team_members' not in st.session_state:
        st.session_state.team_members, _ = load_team_from_csv()

    if 'selected_recipients' not in st.session_state:
        st.session_state.selected_recipients = []

    if 'sender_email' not in st.session_state:
        st.session_state.sender_email = st.session_state.team_members[0]['email'] if st.session_state.team_members else ""

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Team CSV Upload
        st.subheader("📋 Team Data")
        uploaded_file = st.file_uploader("Upload Team CSV (optional)", type=['csv'])

        if uploaded_file is not None:
            team_members, error = load_team_from_csv(uploaded_file)
            if error:
                st.error(error)
            else:
                st.session_state.team_members = team_members
                st.session_state.sender_email = team_members[0]['email'] if team_members else ""
                st.success(f"✅ Loaded {len(team_members)} team members")

        # Show current team stats
        st.metric("Total Team Members", len(st.session_state.team_members))

        if st.button("📊 View Team Details"):
            st.session_state.show_team_details = True

        # Download sample CSV
        if st.button("📥 Download Sample CSV"):
            sample_df = create_sample_team_csv()
            csv = sample_df.to_csv(index=False)
            st.download_button(
                label="Download sample_team.csv",
                data=csv,
                file_name="sample_team.csv",
                mime="text/csv"
            )

        # Memory data editor
        st.subheader("🧠 Memory Data")
        edit_memory = st.checkbox("Edit Memory Data", value=False)

        # Options
        st.subheader("Options")
        send_calendar_invite = st.checkbox("Send calendar invite if meeting required", value=True)

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📧 Email & Calendar Processing")

        # Memory Data Editor (conditional)
        if edit_memory:
            st.subheader("🧠 Edit Memory Data")
            memory_json_str = json.dumps(st.session_state.memory_data, indent=2)
            edited_memory = st.text_area(
                "Memory Data (JSON):",
                value=memory_json_str,
                height=300,
                key="memory_editor"
            )

            if st.button("💾 Update Memory Data"):
                try:
                    st.session_state.memory_data = json.loads(edited_memory)
                    st.success("✅ Memory data updated successfully!")
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON format")
        else:
            # Show current memory data summary
            st.subheader("🧠 Current Memory Summary")
            memories = st.session_state.memory_data.get('results', [])
            st.info(f"📝 {len(memories)} memories loaded")

            # Show key insights
            for i, memory in enumerate(memories[:3]):  # Show first 3
                st.write(f"• {memory.get('memory', '')}")

            if len(memories) > 3:
                st.write(f"... and {len(memories) - 3} more")

        # Team Selection
        st.subheader("👥 Select Email Participants")

# In the team selection section:
        if st.session_state.team_members:
            # Add sender selection dropdown
            st.session_state.sender_email = st.selectbox(
                "Select Email Sender:",
                [member['email'] for member in st.session_state.team_members],
                index=0,
                key="sender_select"
            )

            # Initialize selected_recipients if not exists
            if 'selected_recipients' not in st.session_state:
                st.session_state.selected_recipients = [
                    m['email'] for m in st.session_state.team_members
                    if m['email'] != st.session_state.sender_email
                ]

            # Create checkboxes for team member selection
            all_selected = st.checkbox(
                "Select All Team Members as Recipients",
                value=True,  # Default to all selected
                key="select_all"
            )

            if all_selected:
                st.session_state.selected_recipients = [
                    m['email'] for m in st.session_state.team_members
                    if m['email'] != st.session_state.sender_email
                ]
            else:
                st.write("Select Recipients:")
                temp_selected = []
                for member in st.session_state.team_members:
                    if member['email'] != st.session_state.sender_email:
                        is_selected = st.checkbox(
                            f"{member['name']} ({member['role']}) - {member['email']}",
                            key=f"member_{member['email']}",
                            value=member['email'] in st.session_state.selected_recipients
                        )
                        if is_selected:
                            temp_selected.append(member['email'])
                st.session_state.selected_recipients = temp_selected

            st.info(f"📧 {len(st.session_state.selected_recipients)} recipients selected | Sender: {st.session_state.sender_email}")
        else:
            st.warning("⚠️ No team members loaded. Please upload a CSV file or use sample data.")

        # Process Button
        if st.button("🚀 Process Memory & Draft Email", type="primary", disabled=len(st.session_state.selected_recipients) == 0):
            if st.session_state.selected_recipients:
                try:
                    # Filter selected team members
                    selected_team_members = [
                        member for member in st.session_state.team_members
                        if member['email'] in st.session_state.selected_recipients
                    ]

                    # Show processing status
                    with st.spinner("Processing your request..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        status_text.text("🔍 Analyzing memory data...")
                        progress_bar.progress(25)

                        status_text.text("✍️ Drafting email content...")
                        progress_bar.progress(50)

                        status_text.text("📅 Checking meeting requirements...")
                        progress_bar.progress(75)

                        # Run workflow
                        result = st.session_state.orchestrator.run_workflow(
                            st.session_state.memory_data,
                            st.session_state.team_members,
                            st.session_state.sender_email,
                            st.session_state.selected_recipients
                        )

                        progress_bar.progress(100)
                        status_text.success("✅ Processing complete!")

                    # Display results
                    st.success("🎉 Email and Calendar processing completed!")

                    # Show the LLM prompt
                    with st.expander("🔍 View LLM Prompt Used"):
                        st.write("This is the exact prompt sent to the LLM to generate the email:")
                        st.code(result.get('llm_prompt', 'Prompt not available'), language='text')

                    # Show drafted email
                    st.subheader("📝 Drafted Email")
                    email_content = result.get('email_content', '')

                    # Allow editing of the drafted email
                    edited_email = st.text_area(
                        "Review and edit the email content:",
                        value=email_content,
                        height=300,
                        key="final_email_content"
                    )

                    # Show meeting details if applicable
                    if result.get('meeting_required'):
                        st.subheader("📅 Meeting Details")
                        meeting_details = result.get('meeting_details', {})

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.json(meeting_details)
                        with col_b:
                            st.write("**Meeting Summary:**")
                            st.write(f"📋 Subject: {meeting_details.get('subject', 'N/A')}")
                            st.write(f"📅 Date: {meeting_details.get('date', 'N/A')}")
                            st.write(f"⏰ Time: {meeting_details.get('time', 'N/A')}")
                            st.write(f"⏱️ Duration: {meeting_details.get('duration', 'N/A')}")

                    # Final action buttons
                    st.subheader("🚀 Final Actions")
                    col_send1, col_send2, col_send3 = st.columns(3)

                    with col_send1:
                        if st.button("📧 Send Email Only", type="secondary"):
                            st.info("Email would be sent to selected recipients")
                            st.success("✅ Email sent successfully!")

                    with col_send2:
                        if result.get('meeting_required') and st.button("📅 Send Calendar Only", type="secondary"):
                            st.info("Calendar invite would be sent")
                            st.success("✅ Calendar invite sent!")

                    with col_send3:
                        if st.button("📧📅 Send Both", type="primary"):
                            st.info("Both email and calendar invite would be sent")
                            st.success("✅ Email and calendar invite sent!")

                    # Show execution summary
                    st.subheader("📊 Execution Summary")
                    final_output = result.get('final_output', {})

                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    with col_metric1:
                        st.metric("Recipients", len(selected_team_members))
                    with col_metric2:
                        st.metric("Meeting Required", "Yes" if result.get('meeting_required') else "No")
                    with col_metric3:
                        memories_count = len(st.session_state.memory_data.get('results', []))
                        st.metric("Memories Processed", memories_count)

                    # Show errors if any
                    if result.get('errors'):
                        st.error("⚠️ Errors encountered:")
                        for error in result['errors']:
                            st.write(f"• {error}")

                except Exception as e:
                    st.error(f"❌ Error processing request: {str(e)}")
            else:
                st.warning("⚠️ Please select at least one team member")

    with col2:
        st.header("📋 Current Configuration")

        # Team Details Panel
        if st.session_state.get('show_team_details', False):
            st.subheader("👥 Team Directory")

            if st.session_state.team_members:
                # Search functionality
                search_term = st.text_input("🔍 Search team members:", placeholder="Enter name, role, or department")

                filtered_members = st.session_state.team_members
                if search_term:
                    filtered_members = [
                        member for member in st.session_state.team_members
                        if search_term.lower() in member.get('name', '').lower() or
                           search_term.lower() in member.get('role', '').lower() or
                           search_term.lower() in member.get('department', '').lower()
                    ]

                # Display team members
                for member in filtered_members:
                    with st.expander(f"👤 {member['name']}"):
                        st.write(f"**Role:** {member.get('role', 'N/A')}")
                        st.write(f"**Department:** {member.get('department', 'N/A')}")
                        st.write(f"**Email:** {member.get('email', 'N/A')}")
                        st.write(f"**Location:** {member.get('location', 'N/A')}")
                        st.write(f"**Phone:** {member.get('phone', 'N/A')}")
                        st.write(f"**Project:** {member.get('project', 'N/A')}")

                # Department summary
                if filtered_members:
                    st.subheader("📊 Department Summary")
                    departments = {}
                    for member in filtered_members:
                        dept = member.get('department', 'Unknown')
                        departments[dept] = departments.get(dept, 0) + 1

                    for dept, count in departments.items():
                        st.write(f"• **{dept}**: {count} members")

            if st.button("❌ Close Team Details"):
                st.session_state.show_team_details = False
                st.rerun()

        else:
            # Configuration Summary
            st.subheader("⚙️ Current Settings")

            # Memory summary
            memories = st.session_state.memory_data.get('results', [])
            relations = st.session_state.memory_data.get('relations', {}).get('added_entities', [])

            st.write("**🧠 Memory Data:**")
            st.write(f"• Memories: {len(memories)}")
            st.write(f"• Relations: {len(relations)}")
            st.write(f"• Last Updated: Auto-loaded")

            st.write("**👥 Team Data:**")
            st.write(f"• Total Members: {len(st.session_state.team_members)}")
            st.write(f"• Selected Recipients: {len(st.session_state.selected_recipients)}")
            st.write(f"• Selected Sender: {st.session_state.sender_email}")

            # Show departments
            if st.session_state.team_members:
                departments = set(member.get('department', 'Unknown') for member in st.session_state.team_members)
                st.write(f"• Departments: {', '.join(departments)}")

            st.write("**📧 Options:**")
            st.write(f"• Calendar Invites: {'Enabled' if send_calendar_invite else 'Disabled'}")
            st.write(f"• Memory Editing: {'Enabled' if edit_memory else 'Disabled'}")

            # Recent activity
            st.subheader("🕒 Recent Activities")
            st.write("• Memory data auto-loaded")
            st.write("• Team data loaded from CSV")
            st.write("• System ready for processing")

        # Quick Actions
        st.subheader("⚡ Quick Actions")

        if st.button("🔄 Refresh Team Data"):
            st.session_state.team_members, error = load_team_from_csv()
            if error:
                st.error(error)
            else:
                st.session_state.sender_email = st.session_state.team_members[0]['email'] if st.session_state.team_members else ""
                st.success("✅ Team data refreshed!")

        if st.button("🧠 Reset Memory to Sample"):
            st.session_state.memory_data = SAMPLE_MEMORY_DATA
            st.success("✅ Memory reset to sample data!")

        if st.button("📧 Clear Recipients"):
            st.session_state.selected_recipients = []
            st.success("✅ Recipients cleared!")

        # Tips
        with st.expander("💡 Tips & Help"):
            st.markdown("""
            **Getting Started:**
            1. Upload your team CSV or use sample data
            2. Select email sender and recipients
            3. Edit memory data if needed
            4. Click 'Process Memory & Draft Email'

            **CSV Format:**
            Your CSV should include columns:
            - name, email, role, department, project, location, phone

            **Memory Data:**
            The system auto-loads sample memory data from your Neo4j results.
            You can edit it using the checkbox in the sidebar.

            **Email Participants:**
            - Select who will send the email
            - Select recipients (sender is automatically excluded)
            """)

if __name__ == "__main__":
    create_streamlit_ui()