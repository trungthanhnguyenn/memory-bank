from typing import Dict, Any, Optional
import psycopg2
import json
import logging

from config.postgres_config import PostgresConfig

# Initialize logging
logging.basicConfig(level=logging.INFO)

class PostgresDB:
    """PostgresDB class for interacting with Postgres database"""

    def __init__(self, config: Optional[PostgresConfig] = None):
        """Initialize PostgresDB with configuration"""
        if config is None:
            config = PostgresConfig.from_env()
        self.config = config
        self.url = config.url
        self.user = config.user
        self.password = config.password
        self.host = config.host
        self.port = config.port
        self.max_connections = config.max_connections
        self.socket_timeout = config.socket_timeout

    def connect_postgres(self):
        """Initialize Postgres connection"""
        try:
            conn = psycopg2.connect(
                dbname=self.url,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            return conn
        except Exception as e:
            logging.error(f"Error connecting to Postgres: {e}")
            raise e

    def create_table(self, table_name: str):
        """Create Postgres table for docman information mapping"""
        conn = self.connect_postgres()
        cursor = conn.cursor()
        try:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    session_id VARCHAR(255) NOT NULL,
                    postgres_info VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, session_id)
                )
            """)
            conn.commit()
            logging.info(f"Table {table_name} created successfully")
        except Exception as e:
            logging.error(f"Error creating table {table_name}: {e}")
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()

    def insert_data(self, table_name: str, key_data: Dict[str, str], value_data: Dict[str, str]) -> bool:
        """
        Insert mapping data into Postgres table
        
        Args:
            table_name: Name of the table
            key_data: Dict with user_id and session_id
            value_data: Dict with postgres_info
            
        Returns:
            bool: True if inserted, False if already exists
        """
        conn = self.connect_postgres()
        cursor = conn.cursor()
        try:
            # Check if mapping already exists
            user_id = key_data.get("user_id")
            session_id = key_data.get("session_id")
            postgres_info = value_data.get("postgres_info")
            
            cursor.execute(f"""
                SELECT postgres_info FROM {table_name} 
                WHERE user_id = %s AND session_id = %s
            """, (user_id, session_id))
            
            existing = cursor.fetchone()
            if existing:
                logging.info(f"Mapping already exists for user_id={user_id}, session_id={session_id}")
                return False
            
            # Insert new mapping
            cursor.execute(f"""
                INSERT INTO {table_name} (user_id, session_id, postgres_info)
                VALUES (%s, %s, %s)
            """, (user_id, session_id, postgres_info))
            
            conn.commit()
            logging.info(f"Inserted mapping: user_id={user_id}, session_id={session_id}, postgres_info={postgres_info}")
            return True
            
        except Exception as e:
            logging.error(f"Error inserting data into {table_name}: {e}")
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()

    def select_data(self, table_name: str, user_id: str, session_id: str) -> Optional[str]:
        """
        Select postgres_info for given user_id and session_id
        
        Args:
            table_name: Name of the table
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Optional[str]: postgres_info if found, None otherwise
        """
        conn = self.connect_postgres()
        cursor = conn.cursor()
        try:
            cursor.execute(f"""
                SELECT postgres_info FROM {table_name} 
                WHERE user_id = %s AND session_id = %s
            """, (user_id, session_id))
            
            result = cursor.fetchone()
            if result:
                postgres_info = result[0]
                logging.info(f"Found postgres_info={postgres_info} for user_id={user_id}, session_id={session_id}")
                return postgres_info
            else:
                logging.info(f"No mapping found for user_id={user_id}, session_id={session_id}")
                return None
                
        except Exception as e:
            logging.error(f"Error selecting data from {table_name}: {e}")
            raise e
        finally:
            cursor.close()
            conn.close()

    def delete_data(self, table_name: str, user_id: str, session_id: str) -> bool:
        """
        Delete mapping data from Postgres table
        
        Args:
            table_name: Name of the table
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            bool: True if deleted, False if not found
        """
        conn = self.connect_postgres()
        cursor = conn.cursor()
        try:
            cursor.execute(f"""
                DELETE FROM {table_name} 
                WHERE user_id = %s AND session_id = %s
            """, (user_id, session_id))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            if deleted_count > 0:
                logging.info(f"Deleted mapping for user_id={user_id}, session_id={session_id}")
                return True
            else:
                logging.info(f"No mapping found to delete for user_id={user_id}, session_id={session_id}")
                return False
                
        except Exception as e:
            logging.error(f"Error deleting data from {table_name}: {e}")
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()
