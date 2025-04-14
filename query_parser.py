import pandas as pd
from sql_metadata import Parser
from typing import Dict, Set, List, Optional
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLMetadataExtractor:
    def __init__(self):
        self.source_code_columns = ['AR_SRCE_CDE', 'DATA_SRCE_CDE']
        self.master_dict = {}
        self.alias_map = {}  # Track table aliases
        self.column_map = {}  # Track which columns belong to which tables
        self.reverse_alias_map = {}  # Track reverse mapping of aliases to tables

    def _extract_sql_metadata(self, sql: str) -> Dict[str, List[Set]]:
        """Extract table names, columns, and source codes from SQL query."""
        if not isinstance(sql, str):
            raise ValueError("Input must be a string")
        
        sql = sql.strip()
        if not sql.endswith(';'):
            sql = sql + ';'
        
        try:
            parser = Parser(sql)
            result = {}
            self.alias_map = {}  # Reset maps for new query
            self.column_map = {}
            self.reverse_alias_map = {}
            
            # Get all tables and initialize result dictionary
            tables = parser.tables
            if not tables:
                logger.warning("No tables found in query")
                return {}
            
            # Initialize tables with case-insensitive handling
            for table in tables:
                clean_table = table.split('.')[-1].strip('`"[] ').lower()
                result[clean_table] = [set(), set()]  # [columns, source_codes]
                self.column_map[clean_table] = set()
            
            # Extract table aliases first
            self._extract_table_references(sql)
            
            # Process columns with strict table association
            self._process_columns(parser.columns, result)
            
            # Extract and associate source codes per table
            table_source_codes = self._extract_table_source_codes(sql, tables)
            
            # Add source codes only to their respective tables
            for table, codes in table_source_codes.items():
                clean_table = table.split('.')[-1].strip('`"[] ').lower()
                if clean_table in result:
                    result[clean_table][1].update(codes)
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing SQL: {str(e)}")
            raise

    def _extract_table_references(self, sql: str):
        """Extract table references and their aliases from tables and CTEs."""
        # First, extract CTE names and their aliases
        cte_pattern = r'(?i)WITH\s+(?:RECURSIVE\s+)?(?:.*?,\s*)*(\w+)(?:\s+AS\s+)?\s*\('
        cte_matches = re.finditer(cte_pattern, sql)
        for match in cte_matches:
            cte_name = match.group(1).lower()
            self.alias_map[cte_name] = cte_name  # CTE name is both the table and its own alias
            
        # Match table references in FROM and JOIN clauses, including schema qualified names
        table_pattern = r'(?i)(?:FROM|JOIN)\s+(?:(\w+)\.)?(\w+)(?:\s+(?:AS\s+)?(\w+))?'
        matches = re.finditer(table_pattern, sql)
        
        for match in matches:
            schema, table, alias = match.groups()
            clean_table = table.lower() if table else ''
            
            if alias:
                alias = alias.lower()
                self.alias_map[alias] = clean_table
                self.reverse_alias_map[clean_table] = alias
            else:
                # If no alias, table name can be used as reference
                self.alias_map[clean_table] = clean_table

    def _process_columns(self, columns: List[str], result: Dict[str, List[Set]]):
        """Process columns with strict table association."""
        processed_columns = set()  # Track processed columns to avoid duplicates
        
        # First pass: Process qualified columns
        for column in columns:
            if column == '*' or column.endswith('.*') or 'NULL' in column.upper():
                continue
                
            parts = column.split('.')
            if len(parts) > 1:
                table_ref = parts[-2].strip('`"[] ').lower()
                col = parts[-1].strip('`"[] ').upper()
                
                # Skip if it's NULL
                if col == 'NULL':
                    continue
                
                # Get actual table name from alias if exists
                actual_table = self.alias_map.get(table_ref, table_ref)
                
                if actual_table in result:
                    result[actual_table][0].add(col)
                    self.column_map[actual_table].add(col)
                    processed_columns.add(f"{actual_table}.{col}")

        # Second pass: Process unqualified columns only if they're not already processed
        for column in columns:
            if column == '*' or column.endswith('.*') or 'NULL' in column.upper():
                continue
                
            parts = column.split('.')
            if len(parts) == 1:
                col = parts[0].strip('`"[] ').upper()
                
                # Skip if it's NULL
                if col == 'NULL':
                    continue
                
                # Only add to tables that don't already have this column
                for table in result:
                    if f"{table}.{col}" not in processed_columns:
                        result[table][0].add(col)
                        self.column_map[table].add(col)

    def _extract_table_source_codes(self, sql: str, tables: List[str]) -> Dict[str, Set[str]]:
        """Extract source codes associated with specific tables, handling all reference patterns."""
        table_source_codes = {}
        
        # Initialize source codes for all tables
        for table in tables:
            clean_table = table.split('.')[-1].strip('`"[] ').lower()
            table_source_codes[clean_table] = set()
        
        # Extract source codes using simpler patterns
        for col in self.source_code_columns:
            print(f"\nProcessing column: {col}")
            
            # First find all instances of the column followed by IN clause
            in_pattern = fr'(?i)(\w+)\.{col}\s*(?:\n|\s)*IN\s*\('
            in_matches = re.finditer(in_pattern, sql)
            
            for col_match in in_matches:
                table_ref = col_match.group(1).lower()
                start_pos = col_match.end()
                print(f"\nFound {col} IN reference for table/alias: {table_ref}")
                
                # Find the matching closing parenthesis
                open_count = 1
                pos = start_pos
                while open_count > 0 and pos < len(sql):
                    if sql[pos] == '(':
                        open_count += 1
                    elif sql[pos] == ')':
                        open_count -= 1
                    pos += 1
                
                if open_count == 0:
                    # Extract everything between the parentheses
                    values_str = sql[start_pos:pos-1].strip()
                    print(f"Extracted IN values string: {values_str}")
                    
                    # Split by comma and clean up each value
                    values = set()
                    current_value = ''
                    in_quotes = False
                    
                    for char in values_str:
                        if char == "'":
                            in_quotes = not in_quotes
                            current_value += char
                        elif char == ',' and not in_quotes:
                            if current_value:
                                clean_value = current_value.strip(" '\t\n")
                                if clean_value:
                                    values.add(clean_value.strip("'"))
                                current_value = ''
                        else:
                            current_value += char
                    
                    # Don't forget the last value
                    if current_value:
                        clean_value = current_value.strip(" '\t\n")
                        if clean_value:
                            values.add(clean_value.strip("'"))
                    
                    print(f"Parsed IN values: {values}")
                    
                    # Get actual table name from alias if exists
                    actual_table = self.alias_map.get(table_ref, table_ref)
                    print(f"Actual table name: {actual_table}")
                    
                    # Update source codes for the table
                    if actual_table in table_source_codes:
                        table_source_codes[actual_table].update(values)
                        print(f"Updated source codes for {actual_table}: {table_source_codes[actual_table]}")
            
            # Now handle the equals operator
            equals_pattern = fr'(?i)(\w+)\.{col}\s*(?:\n|\s)*=\s*(?:\n|\s)*\'([^\']+)\''
            equals_matches = re.finditer(equals_pattern, sql)
            
            for eq_match in equals_matches:
                table_ref = eq_match.group(1).lower()
                value = eq_match.group(2)
                print(f"\nFound {col} = reference for table/alias: {table_ref}")
                print(f"Extracted = value: {value}")
                
                # Get actual table name from alias if exists
                actual_table = self.alias_map.get(table_ref, table_ref)
                print(f"Actual table name: {actual_table}")
                
                # Update source codes for the table
                if actual_table in table_source_codes:
                    table_source_codes[actual_table].add(value)
                    print(f"Updated source codes for {actual_table}: {table_source_codes[actual_table]}")
        
        print("\nFinal table source codes:")
        for table, codes in table_source_codes.items():
            print(f"{table}: {codes}")
        
        return table_source_codes

    def _update_master_dict(self, current_metadata: Dict[str, List[Set]]):
        """Update master dictionary with new metadata."""
        for table, (columns, source_codes) in current_metadata.items():
            if table not in self.master_dict:
                self.master_dict[table] = [set(), set()]
            
            self.master_dict[table][0].update(columns)
            self.master_dict[table][1].update(source_codes)

    def process_dataframe(self, df: pd.DataFrame) -> Dict[str, List[Set]]:
        """Process all queries from a DataFrame and build a cumulative metadata dictionary."""
        if 'query_text' not in df.columns:
            raise ValueError("DataFrame must contain 'query_text' column")
        
        for index, row in df.iterrows():
            try:
                query = row['query_text']
                if pd.isna(query) or not query.strip():
                    logger.warning(f"Empty query found at index {index}")
                    continue
                    
                current_metadata = self._extract_sql_metadata(query)
                self._update_master_dict(current_metadata)
                logger.info(f"Successfully processed query at index {index}")
            except Exception as e:
                logger.error(f"Error processing query at index {index}: {str(e)}")
                continue
        
        return self.master_dict

def test_queries():
    """Test the SQL metadata extraction with various cases."""
    extractor = SQLMetadataExtractor()
    
    test_cases = [
        {
            "name": "Source Code Pattern Test",
            "sql": """
                WITH cte1 AS (
                    SELECT t1.col1, t1.data_srce_cde
                    FROM table1 t1
                    WHERE t1.data_srce_cde IN ('A1', 'B2',
                        'C3')
                ),
                cte2 AS (
                    SELECT t2.col1, t2.ar_srce_cde
                    FROM table2 t2
                    WHERE t2.ar_srce_cde = 'X1'
                )
                SELECT c1.*, c2.*
                FROM cte1 c1
                JOIN cte2 c2 ON c1.col1 = c2.col1
                WHERE c2.ar_srce_cde IN ('Y2',
                    'Z3')
            """
        },
        {
            "name": "Mixed Operators Test",
            "sql": """
                SELECT *
                FROM table3 t3
                WHERE t3.data_srce_cde = 'ABC'
                AND t3.ar_srce_cde IN (
                    'DEF',
                    'GHI'
                )
            """
        },
        {
            "name": "Multiple Equals Test",
            "sql": """
                SELECT *
                FROM table4 t4
                WHERE t4.data_srce_cde = 'XYZ'
                UNION ALL
                SELECT *
                FROM table5 t5
                WHERE t5.ar_srce_cde = 
                    'MNO'
            """
        }
    ]
    
    for test in test_cases:
        print(f"\n=== Running Test: {test['name']} ===")
        print("\nSQL Query:")
        print(test['sql'])
        
        try:
            result = extractor._extract_sql_metadata(test['sql'])
            print("\nExtracted Metadata:")
            for table, metadata in result.items():
                print(f"\nTable: {table}")
                print(f"Columns: {sorted(list(metadata[0]))}")
                print(f"Source Codes: {sorted(list(metadata[1]))}")
        except Exception as e:
            print(f"Error processing test case: {str(e)}")
            continue

if __name__ == "__main__":
    test_queries()