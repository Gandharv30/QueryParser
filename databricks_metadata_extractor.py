import pandas as pd
import sqlglot
import sqlglot.expressions as exp
from typing import Dict, Set, List, Optional, Any, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DatabricksMetadataExtractor:
    def __init__(self):
        """
        Initializes the extractor with Databricks dialect.
        """
        self.source_code_columns = {'AR_SRCE_CDE', 'DATA_SRCE_CDE'}
        self.dialect = 'databricks'
        self.master_dict: Dict[str, Dict[str, Set[str]]] = {}
        self.cte_names: Set[str] = set()
        self.table_aliases: Dict[str, str] = {}

    def _process_node(self, node: exp.Expression) -> None:
        """Process a single AST node to extract metadata."""
        logger.debug(f"Processing node: {type(node).__name__}")
        
        # Extract CTEs
        if isinstance(node, exp.With):
            logger.debug("Found WITH node")
            for cte in node.expressions:
                if isinstance(cte, exp.CTE):
                    self.cte_names.add(cte.alias_or_name)
                    logger.debug(f"Added CTE: {cte.alias_or_name}")

        # Extract tables and aliases
        elif isinstance(node, exp.Table):
            table_name = node.name
            if table_name not in self.cte_names and table_name not in self.master_dict:
                self.master_dict[table_name] = {'columns': set(), 'source_codes': set()}
                logger.debug(f"Added table: {table_name}")
                
                # Handle table alias
                if hasattr(node, 'alias') and node.alias:
                    self.table_aliases[node.alias] = table_name
                    logger.debug(f"Added alias {node.alias} for table {table_name}")

        # Extract columns from SELECT expressions
        elif isinstance(node, exp.Select):
            logger.debug("Found SELECT node")
            for expr in node.expressions:
                if isinstance(expr, exp.Column):
                    if hasattr(expr, 'table') and expr.table:
                        alias = expr.table
                        table_name = self.table_aliases.get(alias)
                        if table_name and table_name in self.master_dict:
                            column_name = expr.name
                            self.master_dict[table_name]['columns'].add(column_name)
                            logger.debug(f"Added column {column_name} to table {table_name}")

        # Extract source codes from WHERE conditions
        elif isinstance(node, exp.Where):
            logger.debug("Found WHERE node")
            self._process_where_condition(node.this)

    def _process_where_condition(self, condition: exp.Expression) -> None:
        """Process WHERE conditions to extract source codes."""
        logger.debug(f"Processing WHERE condition: {type(condition).__name__}")
        
        if isinstance(condition, exp.And):
            # Process each part of the AND condition
            logger.debug("Found AND condition")
            for part in condition.expressions:
                self._process_where_condition(part)
            return

        # Handle equality conditions
        if isinstance(condition, exp.EQ):
            logger.debug("Found EQ condition")
            left = condition.left
            right = condition.right
            if (isinstance(left, exp.Column) and 
                left.name in self.source_code_columns and 
                isinstance(right, exp.Literal)):
                alias = left.table
                table_name = self.table_aliases.get(alias)
                if table_name and table_name in self.master_dict:
                    value = right.this.strip("'")
                    self.master_dict[table_name]['source_codes'].add(value)
                    logger.debug(f"Added source code {value} to table {table_name}")

        # Handle IN conditions
        elif isinstance(condition, exp.In):
            logger.debug("Found IN condition")
            left = condition.this
            right = condition.expressions
            if (isinstance(left, exp.Column) and 
                left.name in self.source_code_columns and 
                isinstance(right, list)):
                alias = left.table
                table_name = self.table_aliases.get(alias)
                if table_name and table_name in self.master_dict:
                    values = [val.this.strip("'") for val in right if isinstance(val, exp.Literal)]
                    self.master_dict[table_name]['source_codes'].update(values)
                    logger.debug(f"Added source codes {values} to table {table_name}")

    def _extract_sql_metadata(self, sql_query: str) -> Dict[str, Dict[str, Set[str]]]:
        """Extract metadata from a SQL query using SQLGlot."""
        try:
            logger.debug(f"Processing SQL query: {sql_query}")
            
            # Reset table aliases for each query
            self.table_aliases = {}
            
            # Parse the SQL query into an AST
            ast = sqlglot.parse_one(sql_query, read=self.dialect)
            logger.debug("Successfully parsed SQL query")
            
            # First pass: Extract tables and aliases
            for node in ast.walk():
                if isinstance(node, exp.Table):
                    self._process_node(node)
            
            # Second pass: Extract columns and source codes
            for node in ast.walk():
                if isinstance(node, (exp.Select, exp.Where)):
                    self._process_node(node)
            
            logger.debug(f"Final result: {self.master_dict}")
            return self.master_dict
            
        except Exception as e:
            logger.error(f"Error parsing SQL: {str(e)}")
            return {}

    def process_dataframe(self, df: pd.DataFrame) -> Dict[str, Dict[str, Set[str]]]:
        """Process a DataFrame containing SQL queries and extract metadata."""
        self.master_dict = {}
        self.cte_names = set()
        self.table_aliases = {}
        
        for _, row in df.iterrows():
            try:
                sql_query = row['sql_query']
                if not sql_query or not isinstance(sql_query, str):
                    logger.warning("Empty or invalid SQL query")
                    continue
                    
                self._extract_sql_metadata(sql_query)
                
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                continue
        
        return self.master_dict 