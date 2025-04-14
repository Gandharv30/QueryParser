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
        self.master_dict: Dict[str, Dict[str, List[str]]] = {}
        self.alias_map: Dict[str, str] = {}
        self.cte_names: Set[str] = set()
        self.table_aliases: Dict[str, str] = {}

    def _extract_sql_metadata(self, sql_str: str) -> Dict[str, Dict[str, List[str]]]:
        """
        Extract metadata from SQL query including tables, columns, and source codes.
        """
        if not sql_str or not isinstance(sql_str, str):
            return {}

        # Extract CTEs first to exclude them from results
        ctes = self._extract_ctes(sql_str)
        logging.debug(f"Found CTEs: {ctes}")

        # Extract tables and their aliases
        tables_and_aliases = self._extract_tables_and_aliases(sql_str)
        logging.debug(f"Found tables and aliases: {tables_and_aliases}")

        # Initialize result dictionary
        result = {}

        # Extract columns and source codes for each table
        for table_name, aliases in tables_and_aliases.items():
            # Skip if table is a CTE or contains a CTE name
            if table_name in ctes:
                logging.debug(f"Skipping CTE table: {table_name}")
                continue

            # Skip if table is an alias of another table
            is_alias = any(table_name in alias_list for _, alias_list in tables_and_aliases.items() if _ != table_name)
            if is_alias:
                logging.debug(f"Skipping alias table: {table_name}")
                continue

            # Skip if table name is an alias of a CTE
            is_cte_alias = False
            for cte in ctes:
                if any(table_name == alias for alias in tables_and_aliases.get(cte, [])):
                    is_cte_alias = True
                    logging.debug(f"Skipping CTE alias: {table_name}")
                    break
            if is_cte_alias:
                continue

            # Extract columns and source codes
            columns = self._extract_columns(sql_str, table_name, aliases)
            source_codes = self._extract_source_codes(sql_str, table_name, aliases)

            # Add to result if we found any metadata
            if columns or source_codes:
                result[table_name] = {
                    'columns': columns,
                    'source_codes': source_codes
                }

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
            # Match CTE name before AS, handling various whitespace patterns
            cte_match = re.match(r'^\s*([^\s,(]+)\s+AS\s*\(', cte_def)
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
        """Extract table names and their aliases from SQL query.
        
        Args:
            sql: SQL query string to parse
            
        Returns:
            Dictionary mapping table aliases to their actual table names
        """
        parser = Parser(sql)
        tables = {}

        # First, extract all CTEs to ensure we can properly exclude them
        self._extract_ctes(sql)

        # Process each table and its alias
        for table, alias in parser.tables_aliases.items():
            # Get the actual table name (before the alias)
            table_name = table.split('.')[-1].strip('`"[] ').lower()
            table_alias = alias.split('.')[-1].strip('`"[] ').lower()
            
            # Skip if this is a CTE or a reference to a CTE
            if (table_name in self.cte_names or 
                table_alias in self.cte_names or 
                any(cte in table_name for cte in self.cte_names) or
                any(cte in table_alias for cte in self.cte_names)):
                # Map CTE aliases to their CTE names
                if table_name in self.cte_names:
                    self.alias_map[table_alias] = table_name
                continue

            # Add the actual table name to our list if it's not already there
            if table_name not in tables and table_name not in self.cte_names:
                tables[table_name] = [table_name]
                # Map the alias to the actual table name
                self.alias_map[table_alias] = table_name

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
                any(cte in clean_table for cte in self.cte_names) or
                any(cte in clean_alias for cte in self.cte_names)):
                # Map CTE aliases to their CTE names
                if clean_table in self.cte_names:
                    self.alias_map[clean_alias] = clean_table
                continue
                
            if clean_table not in self.cte_names:
                tables[clean_table] = [clean_alias]
                self.alias_map[clean_alias] = clean_table

        # Filter out any remaining CTEs and their aliases from the tables list
        filtered_tables = {}
        for table, alias_list in tables.items():
            if (table not in self.cte_names and 
                not any(cte in table for cte in self.cte_names) and
                not any(table in cte for cte in self.cte_names) and
                not any(table == alias for alias in self.alias_map.keys() if self.alias_map[alias] in self.cte_names)):
                filtered_tables[table] = alias_list
        
        return filtered_tables

    def _extract_columns(self, sql: str, table_name: str, alias_list: List[str]) -> Set[str]:
        """Extract columns for each table."""
        columns = set()
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
            if actual_table in alias_list:
                columns.add(column)

        return columns

    def _extract_source_codes(self, sql: str, table_name: str, alias_list: List[str]) -> Set[str]:
        """Extract source codes from specific columns."""
        source_codes = set()

        for col in self.source_code_columns:
            # Handle IN clauses
            in_pattern = fr'(?i)(\w+)\.{col}\s*IN\s*\(([^)]+)\)'
            in_matches = re.finditer(in_pattern, sql)
            for match in in_matches:
                table_ref = match.group(1).lower()
                values_str = match.group(2)
                actual_table = self.alias_map.get(table_ref, table_ref)

                if actual_table in alias_list:
                    # Skip if the IN clause contains a SELECT (subquery)
                    if 'SELECT' in values_str.upper():
                        continue
                    # Extract values from IN clause, handling newlines and spaces
                    values = [v.strip(" '\n\t") for v in values_str.split(',') if v.strip()]
                    source_codes.update(values)

            # Handle = operator
            equals_pattern = fr'(?i)(\w+)\.{col}\s*=\s*\'([^\']+)\''
            equals_matches = re.finditer(equals_pattern, sql)
            for match in equals_matches:
                table_ref = match.group(1).lower()
                value = match.group(2)
                actual_table = self.alias_map.get(table_ref, table_ref)

                if actual_table in alias_list:
                    source_codes.add(value.strip())

        return source_codes

    def _update_master_dict(self, current_metadata: Dict[str, Dict[str, List[str]]]):
        """Update master dictionary with new metadata."""
        for table, metadata in current_metadata.items():
            if table not in self.master_dict:
                self.master_dict[table] = {'columns': set(), 'source_codes': set()}
            self.master_dict[table]['columns'].update(metadata['columns'])
            self.master_dict[table]['source_codes'].update(metadata['source_codes'])

    def process_dataframe(self, df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
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
        }
    ]
    
    for test in test_cases:
        try:
            result = extractor._extract_sql_metadata(test['sql'])
            print(f"\n=== Running Test: {test['name']} ===")
            print("\nSQL Query:")
            print(test['sql'])
            print("\nExtracted Metadata:")
            for table, metadata in result.items():
                print(f"\nTable: {table}")
                print(f"Columns: {sorted(list(metadata['columns']))}")
                print(f"Source Codes: {sorted(list(metadata['source_codes']))}")
        except Exception as e:
            print(f"Error processing test case: {str(e)}")
            continue

if __name__ == "__main__":
    test_queries()