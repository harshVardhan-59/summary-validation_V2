import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timedelta
import sqlalchemy as sa
import json
import base64
import os
import streamlit as st
import traceback
import yaml
import logging
import io
import time
from typing import Tuple

# Logger Configuration
logger = logging.getLogger("DBConnection")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Config Loader
def load_config():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, "infra", "vars", "values-dev.yaml")
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        logger.info(f"Config loaded successfully from {config_path}")
        return config

# AWS Secret Manager Helper
def get_secret(secret_name: str, secret_path: str, max_retries: int = 3) -> str:
    region_name = "us-east-1"
    session = boto3.session.Session()
    secrets_manager = session.client(service_name='secretsmanager', region_name=region_name)
    
    for attempt in range(max_retries):
        try:
            response = secrets_manager.get_secret_value(SecretId=secret_path)
            if 'SecretString' in response:
                secret_value = json.loads(response['SecretString']).get(secret_name)
            else:
                secret_value = json.loads(base64.b64decode(response['SecretBinary'])).get(secret_name)
            
            if secret_value is None:
                raise KeyError(f"Secret key '{secret_name}' not found in secret '{secret_path}'")
            
            logger.info(f"Successfully fetched secret '{secret_name}'")
            return secret_value

        except ClientError as e:
            code = e.response['Error']['Code']
            logger.error(f"Attempt {attempt+1}: AWS ClientError [{code}] - {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                raise
        except KeyError as e:
            logger.error(str(e))
            raise

# Database Connection Class
class DatabaseConn:
    def __init__(self):
        try:
            self.DB_USER = "rights_explorer_svc"
            self.DB_PASSWORD = "VgHqUbBZcROKn2yM"
            self.DB_HOST = "localhost"
            self.DB_DATABASE = "rights_explorer"
            self.DB_PORT = 1053

            db_string = f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_DATABASE}"
            
            self.engine_read = sa.create_engine(db_string, pool_size=20, pool_pre_ping=True, pool_recycle=3600)
            self.engine_write = sa.create_engine(db_string, pool_size=20, pool_pre_ping=True, pool_recycle=3600)

            logger.info("Database engines initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize DatabaseConn: {str(e)}")
            traceback.print_exc()
            raise


    def safe_parse(self, value):
        """Try to parse value as JSON, CSV, or return raw."""
        if value is None:
            return None

        if isinstance(value, (dict, list)):
            return value

        if isinstance(value, str):
            v = value.strip()
            if v == "":
                return None
            try:
                return json.loads(v)
            except Exception:
                if "," in v:  # treat comma-separated as list
                    return [x.strip() for x in v.split(",") if x.strip()]
                return v  # fallback: plain string

        return value


    def execute_sql(self, query: str, session_id: str, params: dict = {}, retries: int = 2):
        sanitized_query = query.replace("\n", " ").replace("\r", " ").strip()
        logger.info(f"[DB] Executing SQL | session={session_id} | query: {sanitized_query}")
        timeout_seconds = int(os.environ.get("DB_QUERY_TIMEOUT", 20))

        for attempt in range(retries + 1):
            try:
                with self.engine_read.connect() as conn:
                    conn.execute(sa.text(f"SET statement_timeout = {timeout_seconds * 1000}"))
                    result = conn.execute(sa.text(query), params)
                    columns = result.keys()
                    logger.info(f"[DB] Columns: {list(columns)}")

                    result_list = []
                    for row in result.fetchall():
                        row_dict = {col_name: self.safe_parse(value) for col_name, value in zip(columns, row)}
                        result_list.append(row_dict)

                    logger.info(f"[DB] Query executed successfully | session={session_id} | rows returned: {len(result_list)}")
                    logger.debug(f"[DB] Query response sample: {json.dumps(result_list[:3], indent=2, default=str)}")
                    return result_list

            except Exception as e:
                logger.error(f"[DB] Attempt {attempt+1}: Error executing query | session={session_id} | error={str(e)}")
                if attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise

    
    def download_csv(self, query: str, session_id: str) -> bytes:
        try:
            with self.engine_read.connect() as conn:
                raw_conn = conn.connection.connection
                with raw_conn.cursor() as cur:
                    csv_buffer = io.StringIO()
                    copy_sql = f"COPY ({query}) TO STDOUT WITH CSV HEADER"
                    cur.copy_expert(copy_sql, csv_buffer)
                    logger.info(f"CSV downloaded successfully | session={session_id}")
                    return csv_buffer.getvalue().encode("utf-8")
        except Exception as e:
            logger.error(f"Error downloading CSV | session={session_id} | error={str(e)}")
            raise

    def assume_role(self, role_arn: str, external_id: str) -> Tuple[boto3.Session, datetime]:
        try:
            sts = boto3.client("sts")
            response = sts.assume_role(
                RoleArn=role_arn,
                ExternalId=external_id,
                RoleSessionName="dci-session"
            )
            creds = response['Credentials']
            new_session = boto3.Session(
                aws_access_key_id=creds['AccessKeyId'],
                aws_secret_access_key=creds['SecretAccessKey'],
                aws_session_token=creds['SessionToken']
            )
            expires_at = creds['Expiration']
            logger.info(f"Assumed role successfully, expires at {expires_at}")
            return new_session, expires_at
        except Exception as e:
            logger.error(f"Error assuming role {role_arn}: {str(e)}")
            raise