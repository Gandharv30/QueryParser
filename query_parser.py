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
        
        # Extract source codes using comprehensive patterns
        for col in self.source_code_columns:
            # Main pattern for IN clause - case insensitive, handles newlines
            in_pattern = fr'''(?ix)                 # Case insensitive and verbose mode
                (?:FROM|JOIN|WHERE|AND|OR|\()\s*   # SQL keywords or opening parenthesis
                (?:(\w+)\.)?                       # Optional schema
                (\w+)                              # Table name or alias
                \.{col}                            # Column name
                \s*IN\s*\(\s*                      # IN clause start with optional whitespace
                (                                  # Start capturing values
                    (?:                           # Start non-capturing group
                        '[^']*'                   # A quoted value
                        (?:\s*,\s*'[^']*')*      # More quoted values after commas
                    )                             # End non-capturing group
                )                                 # End capturing values
                \s*\)                             # Closing parenthesis with optional whitespace
            '''
            
            # Pattern for single value with = operator - case insensitive
            equals_pattern = fr'''(?ix)            # Case insensitive and verbose mode
                (?:FROM|JOIN|WHERE|AND|OR|\()\s*  # SQL keywords or opening parenthesis
                (?:(\w+)\.)?                      # Optional schema
                (\w+)                             # Table name or alias
                \.{col}                           # Column name
                \s*=\s*                           # Equals operator with optional whitespace
                '([^']*)'                         # Single quoted value
            '''
            
            # Process IN clause matches
            in_matches = re.finditer(in_pattern, sql)
            for match in in_matches:
                schema, table_ref, values_str = match.groups()
                table_ref = table_ref.lower() if table_ref else ''
                
                # Get actual table name from alias if exists
                actual_table = self.alias_map.get(table_ref, table_ref)
                
                if actual_table in table_source_codes:
                    # Extract all quoted values
                    values = []
                    # Match all quoted values, handling newlines and whitespace
                    quoted_pattern = r"'([^']+)'"
                    quoted_values = re.findall(quoted_pattern, values_str)
                    values.extend([v.strip() for v in quoted_values])
                    
                    # Update source codes for the table
                    table_source_codes[actual_table].update(values)
            
            # Process equals operator matches
            equals_matches = re.finditer(equals_pattern, sql)
            for match in equals_matches:
                schema, table_ref, value = match.groups()
                table_ref = table_ref.lower() if table_ref else ''
                
                # Get actual table name from alias if exists
                actual_table = self.alias_map.get(table_ref, table_ref)
                
                if actual_table in table_source_codes:
                    # Add the single value
                    table_source_codes[actual_table].add(value.strip())
            
            # CTE pattern for IN clause - case insensitive
            cte_in_pattern = fr'''(?ix)            # Case insensitive and verbose mode
                WITH\s+(?:RECURSIVE\s+)?           # WITH or WITH RECURSIVE
                (?:.*?,\s*)*                       # Any preceding CTEs
                (\w+)                              # CTE name
                (?:\s+AS\s+)?\s*\(                # Optional AS and opening parenthesis
                (?:[^()]|\([^()]*\))*?            # Non-greedy match of content
                {col}\s+IN\s*\(\s*                # Column and IN clause
                (                                 # Start capturing values
                    (?:                          # Start non-capturing group
                        '[^']*'                  # A quoted value
                        (?:\s*,\s*'[^']*')*     # More quoted values after commas
                    )                            # End non-capturing group
                )                                # End capturing values
                \s*\)                            # Closing parenthesis
            '''
            
            # Process CTE IN clause matches
            cte_in_matches = re.finditer(cte_in_pattern, sql)
            for match in cte_in_matches:
                cte_name = match.group(1).lower()
                values_str = match.group(2) if len(match.groups()) > 1 else ''
                
                if cte_name in table_source_codes:
                    # Extract values using the same patterns
                    values = []
                    quoted_pattern = r"'([^']+)'"
                    quoted_values = re.findall(quoted_pattern, values_str)
                    values.extend([v.strip() for v in quoted_values])
                    
                    table_source_codes[cte_name].update(values)
        
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
    """Run test cases to verify SQL metadata extraction."""
    
    test_cases = [
        {
            "name": "CTE and Table Reference Variations",
            "query": """
            WITH base_cte AS (
                SELECT 
                    t1.id,
                    t1.DATA_SRCE_CDE,
                    t2.ar_srce_cde
                FROM schema1.table1 t1
                JOIN table2 t2 ON t1.id = t2.id
                WHERE t1.DATA_SRCE_CDE IN ('A1', 'B1')
                AND t2.ar_srce_cde IN ('X1', 'Y1')
            ),
            second_cte AS (
                SELECT 
                    bc.*,
                    t3.DATA_SRCE_CDE
                FROM base_cte bc
                JOIN table3 t3 ON bc.id = t3.id
                WHERE t3.DATA_SRCE_CDE IN ('C1', 'C2')
            )
            SELECT * FROM second_cte;
            """
        },
        {
            "name": "Mixed Case and Reference Styles",
            "query": """
            WITH DataSource AS (
                SELECT 
                    src.id,
                    src.Data_Srce_Cde,
                    tgt.AR_SRCE_CDE
                FROM source_table src
                JOIN target_table tgt ON src.id = tgt.id
                WHERE src.Data_Srce_Cde IN ('D1', 'D2')
                AND tgt.AR_SRCE_CDE IN ('T1', 'T2')
            )
            SELECT 
                ds.*,
                ref.data_srce_cde,
                REF.Ar_Srce_Cde
            FROM DataSource ds
            JOIN reference_table ref ON ds.id = ref.id
            WHERE ref.data_srce_cde IN ('R1', 'R2')
            AND REF.Ar_Srce_Cde IN ('S1', 'S2');
            """
        },
        {
            "name": "Complex CTE Chain with Mixed References",
            "query": """
            WITH first_cte AS (
                SELECT *
                FROM base_table bt
                WHERE bt.DATA_SRCE_CDE IN ('A1', 'B1')
            ),
            SecondCTE AS (
                SELECT 
                    fc.*,
                    t1.ar_srce_cde
                FROM first_cte fc
                JOIN table1 t1 ON fc.id = t1.id
                WHERE t1.ar_srce_cde IN ('X1', 'Y1')
            ),
            ThirdCTE AS (
                SELECT 
                    sc.*,
                    TABLE2.DATA_SRCE_CDE
                FROM SecondCTE sc
                JOIN table2 TABLE2 ON sc.id = TABLE2.id
                WHERE TABLE2.DATA_SRCE_CDE IN ('M1', 'N1')
            )
            SELECT 
                tc.*,
                t3.AR_SRCE_CDE
            FROM ThirdCTE tc
            JOIN table3 t3 ON tc.id = t3.id
            WHERE t3.AR_SRCE_CDE IN ('P1', 'Q1');
            """
        },
        {
            "name": "Subquery and CTE Mixed References",
            "query": """
            WITH source_data AS (
                SELECT *
                FROM (
                    SELECT 
                        t1.id,
                        t1.DATA_SRCE_CDE,
                        t2.ar_srce_cde
                    FROM table1 t1
                    JOIN table2 t2 ON t1.id = t2.id
                    WHERE t1.DATA_SRCE_CDE IN ('A1', 'B1')
                    AND t2.ar_srce_cde IN ('X1', 'Y1')
                ) subq
                WHERE EXISTS (
                    SELECT 1 
                    FROM table3 t3 
                    WHERE t3.id = subq.id
                    AND t3.DATA_SRCE_CDE IN ('C1', 'C2')
                )
            )
            SELECT 
                sd.*,
                t4.AR_SRCE_CDE
            FROM source_data sd
            JOIN table4 t4 ON sd.id = t4.id
            WHERE t4.AR_SRCE_CDE IN ('D1', 'D2');
            """
        },
        {
            "name": "Mixed Operators Test",
            "query": """
            WITH base_data AS (
                SELECT *
                FROM table1 t1
                WHERE t1.DATA_SRCE_CDE = 'A1'
                AND t1.AR_SRCE_CDE IN ('X1', 'Y1')
            ),
            filtered_data AS (
                SELECT 
                    bd.*,
                    t2.data_srce_cde,
                    T2.ar_srce_cde
                FROM base_data bd
                JOIN table2 T2 ON bd.id = T2.id
                WHERE T2.data_srce_cde IN ('B1', 'B2')
                AND T2.ar_srce_cde = 'Z1'
            )
            SELECT 
                fd.*,
                t3.DATA_SRCE_CDE,
                t3.AR_SRCE_CDE
            FROM filtered_data fd
            JOIN table3 t3 ON fd.id = t3.id
            WHERE t3.DATA_SRCE_CDE = 'C1'
            AND t3.AR_SRCE_CDE IN ('W1', 'W2');
            """
        },
        {
            "name": "Case Variation Test",
            "query": """
            WITH SourceData AS (
                SELECT 
                    t1.id,
                    t1.Data_Srce_Cde,
                    t1.AR_SRCE_CDE
                FROM table1 t1
                WHERE t1.Data_Srce_Cde = 'A1'
                AND t1.AR_SRCE_CDE IN ('X1', 'Y1')
            )
            SELECT 
                SD.*,
                t2.data_srce_cde,
                t2.ar_srce_cde
            FROM SourceData SD
            JOIN table2 t2 ON SD.id = t2.id
            WHERE t2.data_srce_cde IN ('B1', 'B2')
            AND t2.ar_srce_cde = 'Z1';
            """
        },
        {
            "name": "Multiple Source Code Values Test",
            "query": """
            SELECT t1.*, t2.column1
            FROM table1 t1
            JOIN table2 t2 ON t1.id = t2.id
            WHERE t1.data_srce_cde IN ('AF', 'BC', 'CA')
            AND t2.ar_srce_cde IN ('X1','Y2', 'Z3')
            AND t2.data_srce_cde = 'D1';
            """
        },
        {
            "name": "Newline IN Clause Test",
            "query": """
            SELECT t1.*, t2.column1
            FROM table1 t1
            JOIN table2 t2 ON t1.id = t2.id
            WHERE 
                t1.data_srce_cde IN ('AF','BC','CA')
            AND t2.ar_srce_cde IN ('X1','Y2', 'Z3')
            AND t2.data_srce_cde = 'D1';
            """
        }
    ]
    
    extractor = SQLMetadataExtractor()
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        print("=" * 80)
        print("Query:")
        print(test_case['query'].strip())
        print("\nResults:")
        
        try:
            result = extractor._extract_sql_metadata(test_case['query'])
            for table, (columns, source_codes) in result.items():
                print(f"\nTable: {table}")
                print("Columns:", sorted(list(columns)))
                print("Source codes:", sorted(list(source_codes)))
        except Exception as e:
            print(f"Error: {str(e)}")
        print("=" * 80)

if __name__ == "__main__":
    test_queries()