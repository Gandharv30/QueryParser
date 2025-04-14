import pandas as pd
from sql_metadata import Parser
from typing import Dict, Set, List, Optional, Tuple
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
        self.cte_names = set()  # Track CTE names to exclude them

    def _extract_sql_metadata(self, sql: str) -> Dict[str, List[Set]]:
        """Extract table names, columns, and source codes from SQL query."""
        print("\n=== Starting SQL metadata extraction ===")
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
            print(f"\nTables found in query: {tables}")
            if not tables:
                logger.warning("No tables found in query")
                return {}
            
            # Initialize tables with case-insensitive handling
            for table in tables:
                clean_table = table.split('.')[-1].strip('`"[] ').lower()
                result[clean_table] = [set(), set()]  # [columns, source_codes]
                self.column_map[clean_table] = set()
                print(f"Initialized result for table: {clean_table}")
            
            # Extract table aliases first
            self._extract_table_references(sql)
            print(f"\nAlias map after extraction: {self.alias_map}")
            
            # Process columns with strict table association
            self._process_columns(parser.columns, result, sql)
            print("\nColumns after processing:")
            for table, (cols, _) in result.items():
                print(f"Table {table} columns: {sorted(list(cols))}")
            
            # Extract and associate source codes per table
            table_source_codes = self._extract_table_source_codes(sql, tables)
            print("\nSource codes after extraction:")
            for table, codes in table_source_codes.items():
                print(f"Table {table} source codes: {sorted(list(codes))}")
            
            # Add source codes only to their respective tables
            for table, codes in table_source_codes.items():
                clean_table = table.split('.')[-1].strip('`"[] ').lower()
                if clean_table in result:
                    # Ensure we're not losing any values during the update
                    print(f"\nUpdating source codes for table {clean_table}")
                    print(f"Current codes in result: {sorted(list(result[clean_table][1]))}")
                    print(f"New codes to add: {sorted(list(codes))}")
                    # Update the source codes set instead of replacing it
                    result[clean_table][1].update(codes)
                    print(f"Updated codes in result: {sorted(list(result[clean_table][1]))}")
            
            print("\nFinal result dictionary:")
            for table, (cols, codes) in result.items():
                print(f"Table: {table}")
                print(f"Columns: {sorted(list(cols))}")
                print(f"Source codes: {sorted(list(codes))}")
            
            return result
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {}

    def _extract_table_references(self, sql: str) -> List[str]:
        """Extract table references and aliases from SQL query."""
        tables = set()  # Use set to avoid duplicates
        self.cte_names = set()  # Track CTE names to exclude them
        
        # Extract CTE names first to exclude them from table list
        cte_pattern = r'(?i)WITH\s+([^\s(]+)'
        cte_matches = re.finditer(cte_pattern, sql)
        for match in cte_matches:
            cte_name = match.group(1).lower()
            self.cte_names.add(cte_name)
            # Add CTE to alias map but not to tables list
            self.alias_map[cte_name] = cte_name
        
        # Extract table references from FROM and JOIN clauses
        table_pattern = r'(?i)(?:FROM|JOIN)\s+([^\s,;()]+)(?:\s+(?:AS\s+)?([^\s,;()]+))?'
        matches = re.finditer(table_pattern, sql)
        
        for match in matches:
            table = match.group(1)
            alias = match.group(2) if match.group(2) else table
            
            # Clean table name and alias
            clean_table = table.split('.')[-1].strip('`"[] ').lower()
            clean_alias = alias.split('.')[-1].strip('`"[] ').lower()
            
            # Skip if this is a CTE or a reference to a CTE
            if clean_table in self.cte_names:
                self.alias_map[clean_alias] = clean_table  # Still track CTE aliases
                continue
            
            # Skip if this table is a CTE reference
            if clean_table in self.alias_map and self.alias_map[clean_table] in self.cte_names:
                continue
            
            # Add to tables list and update maps
            if clean_table not in self.cte_names:  # Double check it's not a CTE
                tables.add(clean_table)
                self.alias_map[clean_alias] = clean_table
                self.reverse_alias_map[clean_table] = clean_alias
                
                # If table is its own alias, add it to alias map
                if clean_table == clean_alias:
                    self.alias_map[clean_table] = clean_table
        
        # Filter out any remaining CTEs from tables
        tables = {t for t in tables if t not in self.cte_names}
        
        return list(tables)

    def _process_columns(self, columns: List[str], result: Dict[str, List[Set]], sql: str):
        """Process columns with strict table association."""
        processed_columns = set()  # Track processed columns to avoid duplicates
        
        for column in columns:
            if '.' not in column:
                continue
                
            try:
                table_ref, column_name = column.split('.')
                table_ref = table_ref.lower()
                column_name = column_name.upper()
                
                if column_name == '*' or 'NULL' in column_name:
                    continue
                    
                # Get actual table name from alias
                actual_table = self.alias_map.get(table_ref, table_ref)
                
                # Only process if it's an actual table and exists in our result
                if actual_table in result:
                    result[actual_table][0].add(column_name)
                    processed_columns.add(column_name)
                    print(f"Added column {column_name} to table {actual_table}")
            except ValueError:
                # Skip columns that don't follow table.column format
                continue
        
        # Also process columns from the original SQL query to catch any missed columns
        column_pattern = r'(?i)(\w+)\.(\w+)'
        matches = re.finditer(column_pattern, sql)
        
        for match in matches:
            table_ref, column = match.groups()
            table_ref = table_ref.lower()
            column = column.upper()
            
            if column == '*' or 'NULL' in column:
                continue
                
            # Get actual table name from alias
            actual_table = self.alias_map.get(table_ref, table_ref)
            
            # Only process if it's an actual table and exists in our result
            if actual_table in result:
                result[actual_table][0].add(column)
                processed_columns.add(column)
                print(f"Added column {column} to table {actual_table} from SQL pattern")

    def _extract_table_source_codes(self, sql: str, tables: List[str]) -> Dict[str, Set]:
        """Extract source codes associated with specific tables, handling all reference patterns."""
        table_source_codes = {}
        
        # Initialize source codes for all tables (only actual tables, not CTEs)
        for table in tables:
            clean_table = table.split('.')[-1].strip('`"[] ').lower()
            if clean_table not in self.cte_names:  # Only add if not a CTE
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
                
                # Get actual table name from alias
                actual_table = self.alias_map.get(table_ref, table_ref)
                
                # Skip if this is a CTE reference or not in our tables list
                if actual_table in self.cte_names or actual_table not in table_source_codes:
                    continue
                    
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
                    print(f"Actual table name: {actual_table}")
                    
                    # Update source codes for the table
                    if actual_table not in self.cte_names:  # Only update if not a CTE
                        table_source_codes[actual_table].update(values)
                        print(f"Updated source codes for {actual_table}: {table_source_codes[actual_table]}")
            
            # Now handle the equals operator
            equals_pattern = fr'(?i)(\w+)\.{col}\s*(?:\n|\s)*=\s*(?:\n|\s)*\'([^\']+)\''
            equals_matches = re.finditer(equals_pattern, sql)
            
            for eq_match in equals_matches:
                table_ref = eq_match.group(1).lower()
                value = eq_match.group(2)
                
                # Get actual table name from alias
                actual_table = self.alias_map.get(table_ref, table_ref)
                
                # Skip if this is a CTE reference or not in our tables list
                if actual_table in self.cte_names or actual_table not in table_source_codes:
                    continue
                    
                print(f"\nFound {col} = reference for table/alias: {table_ref}")
                print(f"Extracted = value: {value}")
                print(f"Actual table name: {actual_table}")
                
                # Update source codes for the table
                if actual_table not in self.cte_names:  # Only update if not a CTE
                    table_source_codes[actual_table].add(value)
                    print(f"Updated source codes for {actual_table}: {table_source_codes[actual_table]}")
        
        print("\nFinal table source codes:")
        for table, codes in table_source_codes.items():
            if table not in self.cte_names:  # Only show actual tables
                print(f"{table}: {codes}")
        
        # Return only non-CTE tables
        return {t: codes for t, codes in table_source_codes.items() if t not in self.cte_names}

    def _update_master_dict(self, current_metadata: Dict[str, List[Set]]):
        """Update master dictionary with new metadata."""
        for table, metadata in current_metadata.items():
            if table not in self.master_dict:
                self.master_dict[table] = [set(), set()]
            
            self.master_dict[table][0].update(metadata[0])  # Update columns
            self.master_dict[table][1].update(metadata[1])  # Update source codes

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
                if table not in extractor.cte_names:  # Only show actual tables
                    print(f"\nTable: {table}")
                    print(f"Columns: {sorted(list(metadata[0]))}")
                    print(f"Source Codes: {sorted(list(metadata[1]))}")
        except Exception as e:
            print(f"Error processing test case: {str(e)}")
            continue

if __name__ == "__main__":
    test_queries()