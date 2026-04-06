import re

# Read the file
with open('orchestrator_main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Change 1: Add user_role and user_name to __init__
old_init = """        self.messages = []
        self.is_active = True
        
    def add_message(self, user_message: str, bot_response: str, bot_type: str):"""

new_init = """        self.messages = []
        self.is_active = True
        self.user_role = None
        self.user_name = None
        
    def add_message(self, user_message: str, bot_response: str, bot_type: str):"""

content = content.replace(old_init, new_init)

# Change 2: Update to_dict() to include user_role and user_name
old_to_dict = """    def to_dict(self) -> Dict:
        return {
            "thread_id": self.thread_id,
            "username": self.username,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages,
            "is_active": self.is_active,
            "message_count": len(self.messages)
        }"""

new_to_dict = """    def to_dict(self) -> Dict:
        return {
            "thread_id": self.thread_id,
            "username": self.username,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages,
            "is_active": self.is_active,
            "message_count": len(self.messages),
            "user_role": self.user_role,
            "user_name": self.user_name
        }"""

content = content.replace(old_to_dict, new_to_dict)

# Change 3: Update thread loading to deserialize user_role and user_name
old_load = """                thread.created_at = thread_data.get("created_at")
                thread.updated_at = thread_data.get("updated_at")
                thread.messages = thread_data.get("messages", [])
                thread.is_active = thread_data.get("is_active", True)
                self.threads[thread_data.get("thread_id")] = thread"""

new_load = """                thread.created_at = thread_data.get("created_at")
                thread.updated_at = thread_data.get("updated_at")
                thread.messages = thread_data.get("messages", [])
                thread.is_active = thread_data.get("is_active", True)
                thread.user_role = thread_data.get("user_role")
                thread.user_name = thread_data.get("user_name")
                self.threads[thread_data.get("thread_id")] = thread"""

content = content.replace(old_load, new_load)

# Change 4: Replace the role checking logic in process_request()
old_role_check = """        # Check if user has set their role, if not prompt for it
        session_info = user_sessions.get(username, {})
        current_role = session_info.get("current_role")
        user_name = session_info.get("name")

        if not current_role:
            # Parse if user provided name and role
            parsed_name, parsed_role = parse_name_and_role(question)
            if parsed_role:
                # User provided role, set it
                asyncio.create_task(asyncio.to_thread(update_user_session, username, parsed_name, parsed_role))
                user_role = parsed_role
                user_name = parsed_name
                logger.info(f"🎭 User set role to: {user_role}, name: {user_name}")
                confirmation = f"Hello {user_name if user_name else username}! I've set your role to {user_role}. How can I help you with GoodBooks ERP today?"
                return {
                    "response": confirmation,
                    "bot_type": "role_setup",
                    "thread_id": thread_id,
                    "user_role": user_role
                }
            else:
                # Ask for name and role
                prompt = \"\"\"Hello! I'm your GoodBooks ERP assistant. To provide you with the best help, please tell me your name and role.

Please reply in this format: "Name: [Your Name], Role: [your role]"

Available roles: developer, implementation, marketing, client, admin, system admin, manager, sales

For example: "Name: John, Role: developer" \"\"\"

                return {
                    "response": prompt,
                    "bot_type": "role_prompt",
                    "thread_id": thread_id,
                    "user_role": "client"  # Default until set
                }

        # Use the stored role
        user_role = current_role

        asyncio.create_task(asyncio.to_thread(update_user_session, username))"""

new_role_check = """        # Check if thread has role set, if not prompt for it
        thread = None
        if thread_id:
            thread = history_manager.get_thread(thread_id)
        
        current_role = thread.user_role if thread else None
        user_name_stored = thread.user_name if thread else None

        if not current_role:
            # Parse if user provided name and role
            parsed_name, parsed_role = parse_name_and_role(question)
            if parsed_role:
                # User provided role, set it in thread
                user_role = parsed_role
                user_name = parsed_name
                if thread:
                    thread.user_role = user_role
                    thread.user_name = user_name
                    history_manager.save_threads()
                logger.info(f"🎭 User set role to: {user_role}, name: {user_name} for thread {thread_id}")
                confirmation = f"Hello {user_name if user_name else username}! I've set your role to {user_role}. How can I help you with GoodBooks ERP today?"
                return {
                    "response": confirmation,
                    "bot_type": "role_setup",
                    "thread_id": thread_id,
                    "user_role": user_role
                }
            else:
                # Ask for name and role
                prompt = \"\"\"Hello! I'm your GoodBooks ERP assistant. To provide you with the best help, please tell me your name and role.

Please reply in this format: "Name: [Your Name], Role: [your role]"

Available roles: developer, implementation, marketing, client, admin, system admin, manager, sales

For example: "Name: John, Role: developer" \"\"\"

                return {
                    "response": prompt,
                    "bot_type": "role_prompt",
                    "thread_id": thread_id,
                    "user_role": "client"  # Default until set
                }

        # Use the stored role from thread
        user_role = current_role"""

content = content.replace(old_role_check, new_role_check)

# Write the updated content
with open('orchestrator_main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Successfully updated orchestrator_main.py with role-per-thread changes")



