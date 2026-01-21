import duckdb
import pandas as pd
from logger import logger
from config import settings

class DuckOrchestrator:
    def __init__(self):
        self.conn = duckdb.connect(database=':memory:')
        self.conn.execute(f"SET memory_limit='{settings.DUCKDB_MEMORY_LIMIT}'")
        self.conn.execute(f"SET threads={settings.DUCKDB_THREADS}")
        self.table_name = "data_view"
        self.current_df = None

    def load_data(self, file_path: str):
        """
        Load data from a CSV or Parquet file into DuckDB.
        """
        logger.info(f"Loading data from {file_path}")
        try:
            if file_path.endswith('.csv'):
                self.conn.execute(f"CREATE OR REPLACE TABLE {self.table_name} AS SELECT * FROM read_csv_auto('{file_path}')")
            elif file_path.endswith('.parquet'):
                self.conn.execute(f"CREATE OR REPLACE TABLE {self.table_name} AS SELECT * FROM read_parquet('{file_path}')")
            else:
                raise ValueError("Unsupported file format. Use CSV or Parquet.")
            
            # Refresh current dataframe view
            self.current_df = self.conn.execute(f"SELECT * FROM {self.table_name}").df()
            logger.info(f"Data loaded successfully. Shape: {self.current_df.shape}")
            return self.current_df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise e

    def transform(self, query: str):
        """
        Apply a SQL transformation to the data.
        If the query is a SELECT statement, updates the view.
        If it's a condition (e.g., "age > 20"), filters the data.
        """
        logger.info(f"Applying transformation: {query}")
        try:
            # Check if it's a raw condition or a full query
            if query.strip().upper().startswith("SELECT"):
                sql = f"CREATE OR REPLACE TABLE {self.table_name} AS {query}"
            else:
                # Assume it's a WHERE clause filter
                sql = f"CREATE OR REPLACE TABLE {self.table_name} AS SELECT * FROM {self.table_name} WHERE {query}"
            
            self.conn.execute(sql)
            self.current_df = self.conn.execute(f"SELECT * FROM {self.table_name}").df()
            logger.info(f"Transformation applied. New shape: {self.current_df.shape}")
            return self.current_df
        except Exception as e:
            logger.error(f"Transformation failed: {e}")
            raise e

    def get_data(self) -> pd.DataFrame:
        """
        Return the current state of the data as a Pandas DataFrame.
        """
        if self.current_df is None:
            # Try to fetch if table exists
            try:
                self.current_df = self.conn.execute(f"SELECT * FROM {self.table_name}").df()
            except:
                logger.warning("No data loaded yet.")
                return pd.DataFrame()
        return self.current_df
        
    def query(self, sql: str) -> pd.DataFrame:
        """
        Run a read-only query and return result.
        """
        try:
            return self.conn.execute(sql).df()
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return pd.DataFrame()
