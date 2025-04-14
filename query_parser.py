import pandas as pd
from sql_metadata import Parser
from typing import Dict, Set, List
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLMetadataExtractor:
    def __init__(self):
        self.source_code_columns = ['AR_SRCE_CDE', 'DATA_SRCE_CDE']
        self.master_dict = {}
        self.alias_map = {}
        self.cte_names = set()

    def _extract_sql_metadata(self, sql: str) -> Dict[str, List[Set]]:
        """Extract table names, columns, and source codes from SQL query."""
        print("\n=== Starting SQL metadata extraction ===")
        if not isinstance(sql, str):
            raise ValueError("Input must be a string")

        sql = sql.strip()
        if not sql.endswith(';'):
            sql = sql + ';'

        try:
            # Reset state for new query
            self.alias_map = {}
            self.cte_names = set()
            result = {}

            # First, extract CTEs to exclude them
            self._extract_ctes(sql)
            print(f"\nCTEs found: {self.cte_names}")

            # Extract tables and their aliases
            tables = self._extract_tables_and_aliases(sql)
            print(f"\nTables found: {tables}")

            # Initialize result dictionary (excluding CTEs)
            for table in tables:
                if table not in self.cte_names:
                    result[table] = [set(), set()]  # [columns, source_codes]
                    print(f"Initialized result for table: {table}")

            # Extract columns for each table
            self._extract_columns(sql, result)
            print("\nColumns after extraction:")
            for table, (cols, _) in result.items():
                print(f"Table {table} columns: {sorted(list(cols))}")

            # Extract source codes
            source_codes = self._extract_source_codes(sql, tables)
            print("\nSource codes after extraction:")
            for table, codes in source_codes.items():
                if table in result and table not in self.cte_names:
                    result[table][1].update(codes)
                    print(f"Table {table} source codes: {sorted(list(codes))}")

            # Final verification to ensure no CTEs are in the result
            result = {k: v for k, v in result.items() if k not in self.cte_names}
            print("\nFinal result after CTE filtering:")
            for table, (cols, codes) in result.items():
                print(f"Table {table}:")
                print(f"  Columns: {sorted(list(cols))}")
                print(f"  Source codes: {sorted(list(codes))}")

            return result

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {}

    def _extract_ctes(self, sql: str):
        """Extract CTE names from WITH clause."""
        # First, find the WITH clause
        with_clause_pattern = r'(?i)WITH\s+(?:RECURSIVE\s+)?([^;]+?)(?=\s+SELECT\s+(?:c1\.|c2\.|[^.]))'
        with_match = re.search(with_clause_pattern, sql, re.DOTALL)
        
        if with_match:
            cte_block = with_match.group(1)
            
            # Pattern to match each CTE definition, handling nested parentheses
            cte_pattern = r'(?i)(\w+)\s+AS\s*\('
            
            # Find all CTE names
            cte_matches = re.finditer(cte_pattern, cte_block)
            for match in cte_matches:
                cte_name = match.group(1).lower().strip()
                self.cte_names.add(cte_name)
                print(f"Found CTE: {cte_name}")
            
            print(f"All CTEs found in WITH clause: {self.cte_names}")
            
            # Find all references to CTEs in the main query
            for cte_name in self.cte_names:
                ref_pattern = fr'(?i)(?:FROM|JOIN)\s+{cte_name}\s+(?:AS\s+)?(\w+)'
                ref_matches = re.finditer(ref_pattern, sql)
                for ref_match in ref_matches:
                    alias = ref_match.group(1).lower().strip()
                    self.alias_map[alias] = cte_name
                    print(f"Found CTE reference: {cte_name} with alias {alias}")

    def _extract_tables_and_aliases(self, sql: str) -> List[str]:
        """Extract table names and their aliases."""
        parser = Parser(sql)
        tables = []

        # First, extract all CTEs to ensure we can properly exclude them
        self._extract_ctes(sql)
        print(f"\nAll CTEs found: {self.cte_names}")

        # Process each table and its alias
        for table, alias in parser.tables_aliases.items():
            # Get the actual table name (before the alias)
            table_name = table.split('.')[-1].strip('`"[] ').lower()
            table_alias = alias.split('.')[-1].strip('`"[] ').lower()
            
            # Skip if this is a CTE or a reference to a CTE
            if table_name in self.cte_names or table_alias in self.cte_names or any(cte in table_name for cte in self.cte_names):
                print(f"Skipping CTE or CTE reference: {table_name} (alias: {table_alias})")
                continue

            # Add the actual table name to our list if it's not already there
            if table_name not in tables:
                tables.append(table_name)
                # Map the alias to the actual table name
                self.alias_map[table_alias] = table_name
                print(f"Added table: {table_name} with alias: {table_alias}")

        # Also extract tables from FROM and JOIN clauses to catch any missed tables
        table_pattern = r'(?i)(?:FROM|JOIN)\s+([^\s,;()]+)(?:\s+(?:AS\s+)?([^\s,;()]+))?'
        matches = re.finditer(table_pattern, sql)
        
        for match in matches:
            table = match.group(1)
            alias = match.group(2) if match.group(2) else table
            
            clean_table = table.split('.')[-1].strip('`"[] ').lower()
            clean_alias = alias.split('.')[-1].strip('`"[] ').lower()
            
            # Skip if this is a CTE or a reference to a CTE or already processed
            if (clean_table in self.cte_names or 
                clean_alias in self.cte_names or 
                clean_table in tables or
                any(cte in clean_table for cte in self.cte_names)):
                print(f"Skipping CTE or duplicate table: {clean_table} (alias: {clean_alias})")
                continue
                
            tables.append(clean_table)
            self.alias_map[clean_alias] = clean_table
            print(f"Added table from FROM/JOIN: {clean_table} with alias: {clean_alias}")

        return [t for t in tables if t not in self.cte_names]

    def _extract_columns(self, sql: str, result: Dict[str, List[Set]]):
        """Extract columns for each table."""
        # Pattern to match table.column references in various contexts
        column_pattern = r'(?i)(?:SELECT|WHERE|ON|AND|OR|,|\(|\s)\s*(\w+)\.(\w+)'
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
                print(f"Added column {column} to table {actual_table}")

    def _extract_source_codes(self, sql: str, tables: List[str]) -> Dict[str, Set]:
        """Extract source codes from specific columns."""
        source_codes = {table: set() for table in tables if table not in self.cte_names}

        for col in self.source_code_columns:
            print(f"\nProcessing column: {col}")

            # Handle IN clauses
            in_pattern = fr'(?i)(\w+)\.{col}\s*IN\s*\(([^)]+)\)'
            in_matches = re.finditer(in_pattern, sql)
            for match in in_matches:
                table_ref = match.group(1).lower()
                values_str = match.group(2)
                actual_table = self.alias_map.get(table_ref, table_ref)

                if actual_table in source_codes:
                    # Extract values from IN clause
                    values = {v.strip(" '") for v in values_str.split(',') if v.strip()}
                    source_codes[actual_table].update(values)
                    print(f"Added IN values for {actual_table}: {values}")

            # Handle = operator
            equals_pattern = fr'(?i)(\w+)\.{col}\s*=\s*\'([^\']+)\''
            equals_matches = re.finditer(equals_pattern, sql)
            for match in equals_matches:
                table_ref = match.group(1).lower()
                value = match.group(2)
                actual_table = self.alias_map.get(table_ref, table_ref)

                if actual_table in source_codes:
                    source_codes[actual_table].add(value)
                    print(f"Added = value for {actual_table}: {value}")

        return source_codes

    def _update_master_dict(self, current_metadata: Dict[str, List[Set]]):
        """Update master dictionary with new metadata."""
        for table, metadata in current_metadata.items():
            if table not in self.master_dict:
                self.master_dict[table] = [set(), set()]
            self.master_dict[table][0].update(metadata[0])  # Update columns
            self.master_dict[table][1].update(metadata[1])  # Update source codes

    def process_dataframe(self, df: pd.DataFrame) -> Dict[str, List[Set]]:
        """Process all queries from a DataFrame."""
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