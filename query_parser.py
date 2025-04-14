import pandas as pd
from sql_metadata import Parser
from typing import Dict, Set, List, Any, Optional
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLMetadataExtractor:
    def __init__(self):
        self.source_code_columns: List[str] = ['AR_SRCE_CDE', 'DATA_SRCE_CDE']
        self.master_dict: Dict[str, Dict[str, Set[str]]] = {}
        self.alias_map: Dict[str, str] = {}
        self.cte_names: Set[str] = set()
        self.table_aliases: Dict[str, str] = {}

    def _extract_sql_metadata(self, sql_str: str) -> Dict[str, Dict[str, Set[str]]]:
        """
        Extract metadata from SQL query including tables, columns, and source codes.
        """
        print("\n=== Starting SQL Metadata Extraction ===")
        logger.info("Starting SQL metadata extraction")
        if not sql_str or not isinstance(sql_str, str):
            logger.warning("Empty or invalid SQL string provided")
            return {}

        # Extract CTEs first to exclude them from results
        self.cte_names = set(self._extract_ctes(sql_str))
        print(f"\nFound CTEs: {self.cte_names}")

        # Extract tables and their aliases
        tables_and_aliases = self._extract_tables_and_aliases(sql_str)
        print(f"\nFound tables and aliases: {tables_and_aliases}")

        # Initialize result dictionary
        result = {}

        # Extract columns and source codes for each table
        for table_name, aliases in tables_and_aliases.items():
            print(f"\nProcessing table: {table_name} with aliases: {aliases}")
            
            # Skip if table is a CTE or contains a CTE name
            if table_name in self.cte_names:
                print(f"Skipping CTE table: {table_name}")
                continue

            # Skip if table is an alias of another table
            is_alias = any(table_name in alias_list for _, alias_list in tables_and_aliases.items() if _ != table_name)
            if is_alias:
                print(f"Skipping alias table: {table_name}")
                continue

            # Skip if table name is an alias of a CTE
            is_cte_alias = False
            for cte in self.cte_names:
                if any(table_name == alias for alias in tables_and_aliases.get(cte, [])):
                    is_cte_alias = True
                    print(f"Skipping CTE alias: {table_name}")
                    break
            if is_cte_alias:
                continue

            # Extract columns and source codes
            columns = self._extract_columns(sql_str, table_name, aliases)
            source_codes = self._extract_source_codes(sql_str, table_name, aliases)
            print(f"Found columns for {table_name}: {columns}")
            print(f"Found source codes for {table_name}: {source_codes}")

            # Add to result if we found any metadata
            if columns or source_codes:
                result[table_name] = {
                    'columns': columns,
                    'source_codes': source_codes
                }
                print(f"Added metadata for table {table_name} to result")

        print(f"\nFinal result dictionary: {result}")
        return result

    def _extract_ctes(self, sql: str) -> List[str]:
        """Extract CTE names from a SQL query.

        Args:
            sql (str): The SQL query to extract CTEs from.

        Returns:
            List[str]: A list of CTE names found in the query.
        """
        logger.info("Starting CTE extraction")
        ctes = []
        cte_hierarchy = {}  # Track parent-child relationships
        
        # First, find the main WITH clause
        with_match = re.search(r'(?i)WITH\s+(?:RECURSIVE\s+)?(.+?)(?=\s+SELECT\s+(?!.*\bAS\s*\())', sql, re.DOTALL)
        if not with_match:
            logger.info("No CTEs found in query")
            return []
        
        cte_block = with_match.group(1)
        logger.info(f"Found CTE block: {cte_block[:100]}...")  # Log first 100 chars for debugging
        
        # Split the CTE block into individual CTEs while respecting nested parentheses
        def split_ctes(cte_block: str) -> List[str]:
            result = []
            current = []
            paren_count = 0
            in_cte = False
            in_subquery = False
            
            for char in cte_block:
                # Track parentheses for subqueries
                if char == '(':
                    paren_count += 1
                    if not in_cte and not in_subquery:
                        in_subquery = True
                elif char == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        in_subquery = False
                
                # Only split at commas when we're at the top level and in a CTE
                if char == ',' and paren_count == 0 and in_cte:
                    if current:
                        result.append(''.join(current).strip())
                        current = []
                    in_cte = False
                    continue
                
                # Check for AS keyword to mark start of CTE definition
                if len(current) >= 2 and ''.join(current[-2:]).upper() == 'AS':
                    in_cte = True
                    in_subquery = False
                
                current.append(char)
            
            if current and in_cte:
                result.append(''.join(current).strip())
            
            return result
        
        cte_definitions = split_ctes(cte_block)
        logger.info(f"Found {len(cte_definitions)} CTE definitions")
        
        # Extract CTE names and build hierarchy
        for cte_def in cte_definitions:
            # Match CTE name before AS, handling various whitespace patterns and column lists
            cte_match = re.match(r'^\s*([^\s,(]+)(?:\s*\([^)]*\))?\s+AS\s*\(', cte_def)
            if cte_match:
                cte_name = cte_match.group(1)
                ctes.append(cte_name)
                logger.info(f"Found CTE: {cte_name}")
                
                # Check for nested CTEs
                nested_with = re.search(r'(?i)WITH\s+(?:RECURSIVE\s+)?(.+?)(?=\s+SELECT\s+(?!.*\bAS\s*\())', cte_def)
                if nested_with:
                    nested_ctes = self._extract_ctes(cte_def)
                    if nested_ctes:
                        cte_hierarchy[cte_name] = nested_ctes
                        ctes.extend(nested_ctes)
                        logger.info(f"Found nested CTEs under {cte_name}: {nested_ctes}")
        
        # Also find any references to CTEs in FROM/JOIN clauses
        main_query = sql[with_match.end():]
        cte_refs = {}  # Map CTEs to their aliases
        for cte in ctes:
            # Look for CTE references in FROM/JOIN clauses
            cte_ref_matches = re.finditer(rf'\b{re.escape(cte)}\s+(?:AS\s+)?([^\s,)]+)', main_query)
            aliases = set()
            for ref in cte_ref_matches:
                alias = ref.group(1)
                aliases.add(alias)
                logger.info(f"Found CTE reference: {cte} AS {alias}")
            if aliases:
                cte_refs[cte] = list(aliases)
        
        # Remove duplicates while preserving order
        ctes = list(dict.fromkeys(ctes))
        logger.info(f"Completed CTE extraction. Found CTEs: {ctes}")
        logger.info(f"CTE hierarchy: {cte_hierarchy}")
        logger.info(f"CTE references: {cte_refs}")
        return ctes

    def _extract_tables_and_aliases(self, sql: str) -> Dict[str, List[str]]:
        """Extract table names and their aliases from SQL query."""
        logger.info("Starting table and alias extraction")
        tables = {}

        # First, extract all CTEs to ensure we can properly exclude them
        self._extract_ctes(sql)

        # Pattern to match table references in FROM and JOIN clauses
        table_pattern = r'(?i)(?:FROM|JOIN)\s+([^\s,;()]+)(?:\s+(?:AS\s+)?([^\s,;()]+))?'
        matches = re.finditer(table_pattern, sql)
        
        for match in matches:
            table = match.group(1)
            alias = match.group(2) if match.group(2) else table
            
            # Clean up table and alias names
            clean_table = table.split('.')[-1].strip('`"[] ').lower()
            clean_alias = alias.split('.')[-1].strip('`"[] ').lower()
            logger.info(f"Found table in FROM/JOIN: {clean_table} with alias: {clean_alias}")
            
            # Skip if this is a CTE or a reference to a CTE
            if (clean_table in self.cte_names or 
                clean_alias in self.cte_names or 
                any(cte in clean_table for cte in self.cte_names) or
                any(cte in clean_alias for cte in self.cte_names)):
                logger.info(f"Skipping CTE or CTE reference: {clean_table} ({clean_alias})")
                # Map CTE aliases to their CTE names
                if clean_table in self.cte_names:
                    self.alias_map[clean_alias] = clean_table
                continue
                
            # Add the table to our list if it's not already there
            if clean_table not in tables:
                tables[clean_table] = [clean_alias]
                # Map the alias to the actual table name
                self.alias_map[clean_alias] = clean_table
                logger.info(f"Added table: {clean_table} with alias: {clean_alias}")

        # Filter out any remaining CTEs and their aliases from the tables list
        filtered_tables = {}
        for table, alias_list in tables.items():
            if (table not in self.cte_names and 
                not any(cte in table for cte in self.cte_names) and
                not any(table in cte for cte in self.cte_names) and
                not any(table == alias for alias in self.alias_map.keys() if self.alias_map[alias] in self.cte_names)):
                filtered_tables[table] = alias_list
                logger.info(f"Kept table after filtering: {table} with aliases: {alias_list}")
            else:
                logger.info(f"Filtered out table: {table}")
        
        logger.info(f"Completed table and alias extraction. Found tables: {filtered_tables}")
        return filtered_tables

    def _extract_columns(self, sql: str, table_name: str, alias_list: List[str]) -> Set[str]:
        """Extract columns for each table."""
        print(f"\nExtracting columns for table: {table_name}")
        print(f"Using aliases: {alias_list}")
        columns = set()
        # Pattern to match table.column references in various contexts
        column_pattern = r'(?i)(?:SELECT|WHERE|ON|AND|OR|,|\(|\s)\s*([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)(?=\s*(?:,|\)|$|\s|AND|OR|IN|>|<|=|!=|>=|<=))'
        matches = re.finditer(column_pattern, sql)

        for match in matches:
            table_ref, column = match.groups()
            table_ref = table_ref.lower()
            column = column.upper()
            print(f"Found potential column reference: {table_ref}.{column}")

            if column == '*' or 'NULL' in column:
                print(f"Skipping wildcard or NULL column: {column}")
                continue

            # Get actual table name from alias
            actual_table = self.alias_map.get(table_ref, table_ref)
            print(f"Resolved table reference: {table_ref} -> {actual_table}")

            # Only process if it's an actual table and exists in our result
            if table_ref in alias_list:
                columns.add(column)
                print(f"Added column {column} for table {actual_table}")

        print(f"Final columns for {table_name}: {columns}")
        return columns

    def _extract_source_codes(self, sql: str, table_name: str, alias_list: List[str]) -> Set[str]:
        """Extract source codes from specific columns."""
        source_codes = set()

        for col in self.source_code_columns:
            # Handle IN clauses with better subquery detection
            in_pattern = fr'(?i)(\w+)\.{col}\s*IN\s*\(([^)]+)\)(?![^()]*\bSELECT\b)'
            in_matches = re.finditer(in_pattern, sql)
            for match in in_matches:
                table_ref = match.group(1).lower()
                values_str = match.group(2)
                actual_table = self.alias_map.get(table_ref, table_ref)

                if actual_table in alias_list:
                    # Skip if the IN clause contains a SELECT (subquery)
                    if 'SELECT' in values_str.upper():
                        continue
                    # Extract values from IN clause, handling newlines, spaces, and nested parentheses
                    values = []
                    current_value = []
                    paren_count = 0
                    for char in values_str:
                        if char == '(':
                            paren_count += 1
                        elif char == ')':
                            paren_count -= 1
                        elif char == ',' and paren_count == 0:
                            if current_value:
                                value = ''.join(current_value).strip(" '\n\t")
                                if value:
                                    values.append(value)
                                current_value = []
                            continue
                        current_value.append(char)
                    
                    if current_value:
                        value = ''.join(current_value).strip(" '\n\t")
                        if value:
                            values.append(value)
                    
                    source_codes.update(values)

            # Handle = operator with better value extraction
            equals_pattern = fr'(?i)(\w+)\.{col}\s*=\s*\'([^\']+)\'(?![^()]*\bSELECT\b)'
            equals_matches = re.finditer(equals_pattern, sql)
            for match in equals_matches:
                table_ref = match.group(1).lower()
                value = match.group(2)
                actual_table = self.alias_map.get(table_ref, table_ref)

                if actual_table in alias_list:
                    source_codes.add(value.strip())

        return source_codes

    def _update_master_dict(self, current_metadata: Dict[str, Dict[str, Set[str]]]):
        """Update master dictionary with new metadata."""
        for table, metadata in current_metadata.items():
            if table not in self.master_dict:
                self.master_dict[table] = {'columns': set(), 'source_codes': set()}
            
            # Convert lists to sets if necessary
            columns = set(metadata['columns']) if isinstance(metadata['columns'], list) else metadata['columns']
            source_codes = set(metadata['source_codes']) if isinstance(metadata['source_codes'], list) else metadata['source_codes']
            
            self.master_dict[table]['columns'].update(columns)
            self.master_dict[table]['source_codes'].update(source_codes)

    def process_dataframe(self, df: pd.DataFrame) -> Dict[str, Dict[str, Set[str]]]:
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
    """Test the SQL metadata extraction with various CTE cases."""
    extractor = SQLMetadataExtractor()
    
    test_cases = [
        {
            "name": "Basic CTE Test",
            "sql": """
                WITH cte1 AS (
                    SELECT t1.col1, t1.data_srce_cde
                    FROM table1 t1
                    WHERE t1.data_srce_cde IN ('A1', 'B2')
                )
                SELECT c1.*
                FROM cte1 c1
            """
        },
        {
            "name": "Multiple CTEs Test",
            "sql": """
                WITH cte1 AS (
                    SELECT t1.col1, t1.data_srce_cde
                    FROM table1 t1
                    WHERE t1.data_srce_cde IN ('A1', 'B2')
                ),
                cte2 AS (
                    SELECT t2.col1, t2.ar_srce_cde
                    FROM table2 t2
                    WHERE t2.ar_srce_cde = 'X1'
                )
                SELECT c1.*, c2.*
                FROM cte1 c1
                JOIN cte2 c2 ON c1.col1 = c2.col1
            """
        },
        {
            "name": "Recursive CTE Test",
            "sql": """
                WITH RECURSIVE cte1 AS (
                    SELECT t1.col1, t1.data_srce_cde
                    FROM table1 t1
                    WHERE t1.data_srce_cde = 'A1'
                    UNION ALL
                    SELECT t1.col1, t1.data_srce_cde
                    FROM table1 t1
                    JOIN cte1 c ON t1.col1 = c.col1
                )
                SELECT c1.*
                FROM cte1 c1
            """
        },
        {
            "name": "Nested CTEs Test",
            "sql": """
                WITH cte1 AS (
                    WITH cte2 AS (
                        SELECT t1.col1, t1.data_srce_cde
                        FROM table1 t1
                        WHERE t1.data_srce_cde = 'A1'
                    )
                    SELECT c2.*
                    FROM cte2 c2
                )
                SELECT c1.*
                FROM cte1 c1
            """
        },
        {
            "name": "CTE with Multiple References Test",
            "sql": """
                WITH cte1 AS (
                    SELECT t1.col1, t1.data_srce_cde
                    FROM table1 t1
                    WHERE t1.data_srce_cde = 'A1'
                )
                SELECT c1.*, c2.*
                FROM cte1 c1
                JOIN cte1 c2 ON c1.col1 = c2.col1
            """
        },
        {
            "name": "CTE with Complex Joins Test",
            "sql": """
                WITH cte1 AS (
                    SELECT t1.col1, t1.data_srce_cde
                    FROM table1 t1
                    WHERE t1.data_srce_cde = 'A1'
                ),
                cte2 AS (
                    SELECT t2.col1, t2.ar_srce_cde
                    FROM table2 t2
                    WHERE t2.ar_srce_cde = 'X1'
                ),
                cte3 AS (
                    SELECT c1.col1, c2.ar_srce_cde
                    FROM cte1 c1
                    JOIN cte2 c2 ON c1.col1 = c2.col1
                )
                SELECT c3.*
                FROM cte3 c3
            """
        },
        {
            "name": "CTE with Subqueries Test",
            "sql": """
                WITH cte1 AS (
                    SELECT t1.col1, t1.data_srce_cde
                    FROM table1 t1
                    WHERE t1.data_srce_cde IN (
                        SELECT t2.data_srce_cde
                        FROM table2 t2
                        WHERE t2.ar_srce_cde = 'X1'
                    )
                )
                SELECT c1.*
                FROM cte1 c1
            """
        },
        {
            "name": "Mega CTE Test with Complex Spacing",
            "sql": """
WITH cte1 AS (
    SELECT t1.col1, t1.data_srce_cde
    FROM table1 t1
    WHERE t1.data_srce_cde IN ('A1', 'B2')
),
	cte2 AS (
		SELECT t2.col1, t2.ar_srce_cde
		FROM table2 t2
		WHERE t2.ar_srce_cde = 'X1'
	)
,
cte3 AS (SELECT t3.col1, t3.data_srce_cde
FROM table3 t3
WHERE t3.data_srce_cde = 'C3'
)
,
	cte4
	AS
	(
		SELECT t4.col1, t4.ar_srce_cde
		FROM table4 t4
		WHERE t4.ar_srce_cde = 'Y2'
	)
,
cte5 AS (
    WITH nested_cte AS (
        SELECT t5.col1, t5.data_srce_cde
        FROM table5 t5
        WHERE t5.data_srce_cde = 'D4'
    )
    SELECT n.*
    FROM nested_cte n
)
,
cte6 AS (
    SELECT t6.col1, t6.ar_srce_cde
    FROM table6 t6
    WHERE t6.ar_srce_cde IN (
        'E5',
        'F6'
    )
)
,
cte7 AS (
    SELECT t7.col1, t7.data_srce_cde
    FROM table7 t7
    WHERE t7.data_srce_cde = 'G7'
)
,
cte8 AS (
    SELECT t8.col1, t8.ar_srce_cde
    FROM table8 t8
    WHERE t8.ar_srce_cde = 'H8'
)
SELECT 
    c1.*,
    c2.*,
    c3.*,
    c4.*,
    c5.*,
    c6.*,
    c7.*,
    c8.*
FROM cte1 c1
JOIN cte2 c2 ON c1.col1 = c2.col1
JOIN cte3 c3 ON c2.col1 = c3.col1
JOIN cte4 c4 ON c3.col1 = c4.col1
JOIN cte5 c5 ON c4.col1 = c5.col1
JOIN cte6 c6 ON c5.col1 = c6.col1
JOIN cte7 c7 ON c6.col1 = c7.col1
JOIN cte8 c8 ON c7.col1 = c8.col1
            """
        },
        {
            "name": "CTE with Column Aliases Test",
            "sql": """
                WITH cte1 AS (
                    SELECT t1.col1 as column1, t1.data_srce_cde as source_code
                    FROM table1 t1
                    WHERE t1.data_srce_cde = 'A1'
                )
                SELECT c1.column1, c1.source_code
                FROM cte1 c1
            """
        },
        {
            "name": "CTE with Multiple Source Codes Test",
            "sql": """
                WITH cte1 AS (
                    SELECT t1.col1, t1.data_srce_cde, t1.ar_srce_cde
                    FROM table1 t1
                    WHERE t1.data_srce_cde = 'A1'
                    AND t1.ar_srce_cde IN ('X1', 'X2', 'X3')
                )
                SELECT c1.*
                FROM cte1 c1
            """
        },
        {
            "name": "CTE with Complex WHERE Conditions",
            "sql": """
                WITH cte1 AS (
                    SELECT t1.col1, t1.data_srce_cde
                    FROM table1 t1
                    WHERE t1.data_srce_cde = 'A1'
                    AND (t1.col1 > 100 OR t1.col1 < 50)
                    AND t1.col2 IS NOT NULL
                )
                SELECT c1.*
                FROM cte1 c1
            """
        },
        {
            "name": "CTE with UNION and Multiple Tables",
            "sql": """
                WITH cte1 AS (
                    SELECT t1.col1, t1.data_srce_cde
                    FROM table1 t1
                    WHERE t1.data_srce_cde = 'A1'
                    UNION ALL
                    SELECT t2.col1, t2.ar_srce_cde
                    FROM table2 t2
                    WHERE t2.ar_srce_cde = 'X1'
                )
                SELECT c1.*
                FROM cte1 c1
            """
        },
        {
            "name": "CTE with Self-Join and Window Functions",
            "sql": """
                WITH cte1 AS (
                    SELECT 
                        t1.col1,
                        t1.data_srce_cde,
                        ROW_NUMBER() OVER (PARTITION BY t1.data_srce_cde ORDER BY t1.col1) as rn
                    FROM table1 t1
                ),
                cte2 AS (
                    SELECT 
                        c1.col1,
                        c1.data_srce_cde,
                        c2.col1 as prev_col1
                    FROM cte1 c1
                    LEFT JOIN cte1 c2 ON c1.rn = c2.rn + 1
                        AND c1.data_srce_cde = c2.data_srce_cde
                )
                SELECT c2.*
                FROM cte2 c2
            """
        },
        {
            "name": "CTE with Nested Subqueries in JOIN",
            "sql": """
                WITH cte1 AS (
                    SELECT t1.col1, t1.data_srce_cde
                    FROM table1 t1
                    WHERE t1.data_srce_cde = 'A1'
                )
                SELECT c1.*, t2.*
                FROM cte1 c1
                JOIN table2 t2 ON t2.col1 = c1.col1
                    AND t2.ar_srce_cde IN (
                        SELECT t3.ar_srce_cde
                        FROM table3 t3
                        WHERE t3.data_srce_cde = 'B1'
                    )
            """
        },
        {
            "name": "CTE with Complex Column References",
            "sql": """
                WITH cte1 AS (
                    SELECT 
                        CASE 
                            WHEN t1.data_srce_cde = 'A1' THEN t1.col1
                            WHEN t1.data_srce_cde = 'B1' THEN t2.col1
                            ELSE t3.col1
                        END as derived_col,
                        t1.data_srce_cde,
                        t2.ar_srce_cde,
                        t3.data_srce_cde as alt_source
                    FROM table1 t1
                    LEFT JOIN table2 t2 ON t1.col1 = t2.col1
                    LEFT JOIN table3 t3 ON t2.col1 = t3.col1
                )
                SELECT c1.*
                FROM cte1 c1
            """
        }
    ]
    
    for test in test_cases:
        try:
            print(f"\n=== Running Test: {test['name']} ===")
            print("\nSQL Query:")
            print(test['sql'])
            result = extractor._extract_sql_metadata(test['sql'])
            print("\nExtracted Metadata:")
            for table, metadata in result.items():
                print(f"\nTable: {table}")
                print(f"Columns: {sorted(list(metadata['columns']))}")
                print(f"Source Codes: {sorted(list(metadata['source_codes']))}")
        except Exception as e:
            print(f"Error processing test case {test['name']}: {str(e)}")
            continue

if __name__ == "__main__":
    test_queries()