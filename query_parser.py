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
        
        # Remove comments first to avoid false matches
        def remove_comments(sql):
            # Remove inline comments
            sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
            # Remove block comments
            sql = re.sub(r'/\*[\s\S]*?\*/', '', sql)
            return sql
        
        sql = remove_comments(sql)
        
        # First, find the main WITH clause, handling complex whitespace and comments
        with_match = re.search(
            r'(?i)WITH\s*(?:RECURSIVE\s+)?'  # WITH keyword with optional RECURSIVE
            r'(?:\s*--[^\n]*\n)*'            # Optional inline comments
            r'(?:\s*/\*[\s\S]*?\*/)*'        # Optional block comments
            r'\s*(.+?)'                       # CTE content
            r'(?=\s+SELECT\s+(?!.*\bAS\s*\())',  # Look ahead for SELECT
            sql, 
            re.DOTALL
        )
        
        if not with_match:
            logger.info("No CTEs found in query")
            return []
        
        cte_block = with_match.group(1)
        logger.info(f"Found CTE block: {cte_block[:100]}...")
        
        def split_ctes(cte_block: str) -> List[str]:
            result = []
            current = []
            paren_count = 0
            quote_char = None
            in_cte = False
            in_subquery = False
            
            i = 0
            while i < len(cte_block):
                char = cte_block[i]
                
                # Handle quoted identifiers and string literals
                if char in ('"', '`', "'") and (i == 0 or cte_block[i-1] != '\\'):
                    if quote_char is None:
                        quote_char = char
                    elif char == quote_char:
                        quote_char = None
                    current.append(char)
                    i += 1
                    continue
                
                # Skip characters in quotes
                if quote_char is not None:
                    current.append(char)
                    i += 1
                    continue
                
                # Track parentheses for subqueries
                if char == '(':
                    paren_count += 1
                    if not in_cte and not in_subquery:
                        in_subquery = True
                elif char == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        in_subquery = False
                
                # Handle commas at the top level
                if char == ',' and paren_count == 0 and in_cte:
                    if current:
                        result.append(''.join(current).strip())
                        current = []
                    in_cte = False
                    i += 1
                    continue
                
                # Look for AS keyword
                if len(current) >= 2 and not in_cte:
                    last_two = ''.join(current[-2:]).upper()
                    if last_two == 'AS' and not re.search(r'[a-zA-Z0-9_]', cte_block[i+1:i+2] or ''):
                        in_cte = True
                        in_subquery = False
                
                current.append(char)
                i += 1
            
            if current and in_cte:
                result.append(''.join(current).strip())
            
            return result
        
        cte_definitions = split_ctes(cte_block)
        logger.info(f"Found {len(cte_definitions)} CTE definitions")
        
        # Enhanced pattern for CTE name extraction
        cte_name_pattern = r'''
            ^\s*                           # Start of string with optional whitespace
            (?:["`]([^"`]+)["`]|          # Quoted identifier
            ([a-zA-Z0-9_-]+))             # Regular identifier
            \s*                           # Optional whitespace
            (?:\s*\([^)]*\))?            # Optional column list
            \s+AS\s*\(                    # AS keyword and opening parenthesis
        '''
        
        # Extract CTE names and build hierarchy
        for cte_def in cte_definitions:
            # Match CTE name with enhanced pattern
            cte_match = re.match(cte_name_pattern, cte_def, re.VERBOSE)
            if cte_match:
                # Get the CTE name from either quoted or unquoted group
                cte_name = (cte_match.group(1) or cte_match.group(2)).strip('`"')
                ctes.append(cte_name)
                logger.info(f"Found CTE: {cte_name}")
                
                # Check for nested CTEs
                nested_with = re.search(
                    r'(?i)WITH\s*(?:RECURSIVE\s+)?'
                    r'(?:\s*--[^\n]*\n)*'
                    r'(?:\s*/\*[\s\S]*?\*/)*'
                    r'\s*(.+?)'
                    r'(?=\s+SELECT\s+(?!.*\bAS\s*\())',
                    cte_def,
                    re.DOTALL
                )
                if nested_with:
                    nested_ctes = self._extract_ctes(cte_def)
                    if nested_ctes:
                        cte_hierarchy[cte_name] = nested_ctes
                        ctes.extend(nested_ctes)
                        logger.info(f"Found nested CTEs under {cte_name}: {nested_ctes}")
        
        # Find CTE references with enhanced pattern
        main_query = sql[with_match.end():]
        cte_refs = {}
        for cte in ctes:
            # Look for CTE references with quoted identifiers and aliases
            cte_ref_pattern = fr'''
                \b{re.escape(cte)}\b  # CTE name
                \s*                    # Optional whitespace
                (?:AS\s+)?            # Optional AS keyword
                (?:                    # Start alias group
                    ["`]([^"`]+)["`]| # Quoted alias
                    ([a-zA-Z0-9_-]+)  # Regular alias
                )
            '''
            cte_ref_matches = re.finditer(cte_ref_pattern, main_query, re.VERBOSE)
            aliases = set()
            for ref in cte_ref_matches:
                # Get alias from either quoted or unquoted group
                alias = (ref.group(1) or ref.group(2)).strip('`"')
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

    def _extract_tables_and_aliases(self, query: str) -> Dict[str, List[str]]:
        """Extract tables and their aliases from the query."""
        tables = {}
        self.alias_map = {}
        
        # First extract CTEs to avoid processing them as tables
        ctes = self._extract_ctes(query)
        self.cte_names = set(ctes)
        
        # Original pattern for basic table identification
        table_pattern = r'(?i)(?:FROM|JOIN)\s+([^\s,;()]+)(?:\s+(?:AS\s+)?([^\s,;()]+))?'
        
        # Enhanced patterns for edge cases
        additional_patterns = [
            # Pattern for schema-qualified tables with optional quotes and complex identifiers
            r'''(?ix)
                (?:FROM|JOIN)\s+
                (?:
                    (?:["`]?(?:[a-zA-Z0-9_-]+(?:\s*\.\s*[a-zA-Z0-9_-]+)*)["`]?\.)? # Optional schema
                    ["`]?([a-zA-Z0-9_-]+(?:\s*\.\s*[a-zA-Z0-9_-]+)*)["`]?          # Table name
                    (?:\s+(?:AS\s+)?["`]?([a-zA-Z0-9_-]+)["`]?)?                    # Optional alias
                )
            ''',
            
            # Pattern for tables after newlines/whitespace with comments
            r'''(?ix)
                (?:FROM|JOIN)\s*
                (?:--[^\n]*\n|/\*(?:[^*]|\*(?!/))*\*/)*   # Optional comments
                \s*
                ["`]?([a-zA-Z0-9_-]+(?:\s*\.\s*[a-zA-Z0-9_-]+)*)["`]?
                (?:\s+(?:AS\s+)?["`]?([a-zA-Z0-9_-]+)["`]?)?
            ''',
            
            # Pattern for tables in subqueries with complex whitespace and comments
            r'''(?ix)
                (?:FROM|JOIN)\s*\(\s*
                (?:SELECT|WITH)[\s\S]*?
                FROM\s+
                ["`]?([a-zA-Z0-9_-]+(?:\s*\.\s*[a-zA-Z0-9_-]+)*)["`]?
                (?:\s+(?:AS\s+)?["`]?([a-zA-Z0-9_-]+)["`]?)?
            ''',
            
            # Pattern for tables in PIVOT/UNPIVOT operations
            r'''(?ix)
                (?:PIVOT|UNPIVOT)\s*\(\s*
                [^)]*\)\s+(?:AS\s+)?
                ["`]?([a-zA-Z0-9_-]+)["`]?
                (?:\s+(?:AS\s+)?["`]?([a-zA-Z0-9_-]+)["`]?)?
            ''',
            
            # Pattern for tables in CROSS/OUTER APPLY with table-valued functions
            r'''(?ix)
                (?:CROSS|OUTER)\s+APPLY\s+
                ["`]?([a-zA-Z0-9_-]+(?:\s*\.\s*[a-zA-Z0-9_-]+)*)["`]?
                (?:\s*\([^)]*\))?
                (?:\s+(?:AS\s+)?["`]?([a-zA-Z0-9_-]+)["`]?)?
            ''',
            
            # Pattern for tables in MERGE statements
            r'''(?ix)
                MERGE(?:\s+INTO)?\s+
                ["`]?([a-zA-Z0-9_-]+(?:\s*\.\s*[a-zA-Z0-9_-]+)*)["`]?
                (?:\s+(?:AS\s+)?["`]?([a-zA-Z0-9_-]+)["`]?)?
            ''',
            
            # Pattern for tables in complex JOIN variations
            r'''(?ix)
                (?:
                    (?:LEFT|RIGHT|FULL|INNER|CROSS|NATURAL)\s+
                    (?:OUTER\s+)?
                    (?:HASH\s+)?
                    (?:LOOP\s+)?
                )?
                JOIN\s+
                ["`]?([a-zA-Z0-9_-]+(?:\s*\.\s*[a-zA-Z0-9_-]+)*)["`]?
                (?:\s+(?:AS\s+)?["`]?([a-zA-Z0-9_-]+)["`]?)?
            ''',
            
            # Pattern for tables in USING clause with complex identifiers
            r'''(?ix)
                USING\s+
                ["`]?([a-zA-Z0-9_-]+(?:\s*\.\s*[a-zA-Z0-9_-]+)*)["`]?
                (?:\s+(?:AS\s+)?["`]?([a-zA-Z0-9_-]+)["`]?)?
            ''',
            
            # Pattern for tables in LATERAL joins with subqueries
            r'''(?ix)
                LATERAL\s+
                (?:
                    (?:SELECT|WITH)[\s\S]*?FROM\s+
                    ["`]?([a-zA-Z0-9_-]+(?:\s*\.\s*[a-zA-Z0-9_-]+)*)["`]?
                    |
                    ["`]?([a-zA-Z0-9_-]+(?:\s*\.\s*[a-zA-Z0-9_-]+)*)["`]?
                )
                (?:\s+(?:AS\s+)?["`]?([a-zA-Z0-9_-]+)["`]?)?
            '''
        ]
        
        def process_matches(pattern, query):
            matches = re.finditer(pattern, query)
            for match in matches:
                groups = match.groups()
                
                # Handle schema-qualified tables
                if len(groups) == 3 and groups[0]:  # Schema present
                    schema, table_name, alias = groups
                    table_name = f"{schema}.{table_name}"
                else:
                    # Get last two groups for table and alias
                    table_name = groups[-2] if len(groups) >= 2 else groups[0]
                    alias = groups[-1] if len(groups) >= 2 and groups[-1] else table_name
                
                if not table_name:
                    continue
                    
                table_name = table_name.lower()
                alias = alias.lower() if alias else table_name
                
                # Skip CTEs and their aliases
                if table_name in self.cte_names or alias in self.cte_names:
                    continue
                    
                # Handle quoted identifiers
                table_name = table_name.strip('`"')
                alias = alias.strip('`"')
                
                if table_name not in tables:
                    tables[table_name] = []
                if alias not in tables[table_name]:
                    tables[table_name].append(alias)
                self.alias_map[alias] = table_name
                logging.info(f"Found table: {table_name} with alias: {alias}")
        
        # Process original pattern first (preserved)
        process_matches(table_pattern, query)
        
        # Process additional patterns
        for pattern in additional_patterns:
            process_matches(pattern, query)
        
        # Filter out any remaining CTEs or CTE references
        filtered_tables = {}
        for table_name, aliases in tables.items():
            if table_name not in self.cte_names and not any(alias in self.cte_names for alias in aliases):
                filtered_tables[table_name] = aliases
                logging.info(f"Kept table after filtering: {table_name} with aliases: {aliases}")
        
        return filtered_tables

    def _extract_columns(self, sql: str, table_name: str, alias_list: List[str]) -> Set[str]:
        """Extract columns for each table."""
        print(f"\nExtracting columns for table: {table_name}")
        print(f"Using aliases: {alias_list}")
        columns = set()
        
        # Original pattern for basic column references (preserved)
        column_pattern = r'(?i)(?:SELECT|WHERE|ON|AND|OR|,|\(|\s)\s*([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)(?=\s*(?:,|\)|$|\s|AND|OR|IN|>|<|=|!=|>=|<=))'
        
        # Enhanced patterns for edge cases
        additional_patterns = [
            # Pattern for columns in window functions with complex expressions and partitioning
            r'''(?ix)
                (?:
                    PARTITION\s+BY|
                    ORDER\s+BY|
                    ROWS|RANGE\s+BETWEEN
                )\s+
                (?:["`]?([a-zA-Z0-9_-]+)["`]?\.)
                (?:["`]?([a-zA-Z0-9_-]+)["`]?)
                (?:\s*(?:ASC|DESC|NULLS\s+(?:FIRST|LAST))?)?
            ''',
            
            # Pattern for columns in aggregate functions with DISTINCT and nested expressions
            r'''(?ix)
                (?:
                    COUNT|SUM|AVG|MAX|MIN|STDDEV|VARIANCE|
                    FIRST_VALUE|LAST_VALUE|LAG|LEAD|
                    STRING_AGG|ARRAY_AGG
                )\s*\(\s*
                (?:DISTINCT\s+)?
                (?:ALL\s+)?
                (?:["`]?([a-zA-Z0-9_-]+)["`]?\.)
                (?:["`]?([a-zA-Z0-9_-]+)["`]?)
                (?:\s*(?:,\s*\d+|\s*IGNORE\s+NULLS)?)?
                \s*\)
            ''',
            
            # Pattern for columns in CASE statements with multiple conditions and nested expressions
            r'''(?ix)
                (?:
                    CASE\s+(?:["`]?([a-zA-Z0-9_-]+)["`]?\.)(?:["`]?([a-zA-Z0-9_-]+)["`]?)|
                    WHEN\s+(?:["`]?([a-zA-Z0-9_-]+)["`]?\.)(?:["`]?([a-zA-Z0-9_-]+)["`]?)|
                    THEN\s+(?:["`]?([a-zA-Z0-9_-]+)["`]?\.)(?:["`]?([a-zA-Z0-9_-]+)["`]?)|
                    ELSE\s+(?:["`]?([a-zA-Z0-9_-]+)["`]?\.)(?:["`]?([a-zA-Z0-9_-]+)["`]?)
                )
                (?:\s*(?:[=<>!]+|\bIS\b|\bLIKE\b|\bIN\b|\bBETWEEN\b)\s*)?
            ''',
            
            # Pattern for columns with special characters, quoted identifiers, and schema qualification
            r'''(?ix)
                (?:SELECT|WHERE|ON|AND|OR|,|\(|\s)\s*
                (?:["`]?(?:[a-zA-Z0-9_-]+\.)?[a-zA-Z0-9_-]+["`]?\.)
                ["`]?([a-zA-Z0-9_-]+)["`]?
                (?:\s*(?:AS\s+["`]?[a-zA-Z0-9_-]+["`]?)?)?
            ''',
            
            # Pattern for columns in complex expressions and functions with type casting
            r'''(?ix)
                (?:
                    COALESCE|NULLIF|CAST|CONVERT|TRY_CAST|TRY_CONVERT|
                    ISNULL|NVL|DECODE|IFF
                )\s*\(\s*
                (?:["`]?([a-zA-Z0-9_-]+)["`]?\.)
                (?:["`]?([a-zA-Z0-9_-]+)["`]?)
                (?:\s*,|\s+AS\b)?
            ''',
            
            # Pattern for columns in subqueries and EXISTS clauses with complex conditions
            r'''(?ix)
                (?:EXISTS|IN|ANY|ALL|SOME)\s*\(\s*
                SELECT\b(?:(?!SELECT|FROM).)*?
                (?:["`]?([a-zA-Z0-9_-]+)["`]?\.)
                (?:["`]?([a-zA-Z0-9_-]+)["`]?)
            ''',
            
            # Pattern for columns in GROUP BY and HAVING clauses with expressions
            r'''(?ix)
                (?:GROUP\s+BY|HAVING)\s+
                (?:["`]?([a-zA-Z0-9_-]+)["`]?\.)
                (?:["`]?([a-zA-Z0-9_-]+)["`]?)
                (?:\s*,|\s+(?:ASC|DESC)|\s*(?:[=<>!]+|\bIS\b|\bLIKE\b|\bIN\b|\bBETWEEN\b))?
            ''',
            
            # Pattern for columns in JSON/XML path expressions
            r'''(?ix)
                (?:
                    JSON_VALUE|JSON_QUERY|JSON_MODIFY|
                    XPATH|XQUERY
                )\s*\(\s*
                (?:["`]?([a-zA-Z0-9_-]+)["`]?\.)
                (?:["`]?([a-zA-Z0-9_-]+)["`]?)
                \s*,
            ''',
            
            # Pattern for columns in MERGE statements with complex conditions
            r'''(?ix)
                (?:
                    MERGE\s+INTO\s+[^\s]+\s+(?:AS\s+)?[^\s]+\s+
                    (?:USING|ON|WHEN\s+MATCHED|WHEN\s+NOT\s+MATCHED)\s+
                )?
                (?:["`]?([a-zA-Z0-9_-]+)["`]?\.)
                (?:["`]?([a-zA-Z0-9_-]+)["`]?)
            ''',
            
            # Pattern for columns in PIVOT/UNPIVOT operations
            r'''(?ix)
                (?:PIVOT|UNPIVOT)\s*\(\s*
                (?:
                    [^\s]*\s+
                    (?:["`]?([a-zA-Z0-9_-]+)["`]?\.)
                    (?:["`]?([a-zA-Z0-9_-]+)["`]?)
                    |
                    (?:["`]?([a-zA-Z0-9_-]+)["`]?\.)
                    (?:["`]?([a-zA-Z0-9_-]+)["`]?)
                    \s+[^\s]*
                )
            '''
        ]
        
        def process_matches(pattern, sql):
            matches = re.finditer(pattern, sql)
            for match in matches:
                groups = match.groups()
                
                # Process all potential table/column pairs in the groups
                for i in range(0, len(groups), 2):
                    if i + 1 >= len(groups) or not groups[i] or not groups[i+1]:
                        continue
                        
                    table_ref = groups[i].lower()
                    column = groups[i+1].upper()
                    
                    if column == '*' or 'NULL' in column:
                        continue
                    
                    # Handle quoted identifiers
                    table_ref = table_ref.strip('`"')
                    column = column.strip('`"')
                    
                    # Only process if it's a relevant table reference
                    if table_ref in alias_list:
                        columns.add(column)
                        print(f"Added column {column} for table {table_name} via {table_ref}")
        
        # Process original pattern first (preserved)
        process_matches(column_pattern, sql)
        
        # Process additional patterns
        for pattern in additional_patterns:
            process_matches(pattern, sql)
        
        print(f"Final columns for {table_name}: {columns}")
        return columns

    def _extract_source_codes(self, query: str, table_name: str, alias_list: List[str]) -> Set[str]:
        """Extract source codes from the query for a specific table."""
        source_codes = set()
        print(f"\nExtracting source codes for table: {table_name}")
        print(f"Using aliases: {alias_list}")
        
        # Original pattern for basic source code values (preserved)
        source_code_pattern = r'(?i)(?:WHERE|AND|OR|ON|WHEN)\s+(?:' + '|'.join(alias_list) + r')\.(?:data_srce_cde|ar_srce_cde)\s*=\s*[\'"]([^\'"]+)[\'"]'
        
        # Enhanced patterns for edge cases
        additional_patterns = [
            # Pattern for source codes in CASE statements with multiple conditions
            r'(?i)(?:CASE\s+(?:' + '|'.join(alias_list) + r')\.(?:data_srce_cde|ar_srce_cde)\s+WHEN\s+[\'"]([^\'"]+)[\'"]|'
            r'WHEN\s+(?:' + '|'.join(alias_list) + r')\.(?:data_srce_cde|ar_srce_cde)\s*=\s*[\'"]([^\'"]+)[\'"])',
            
            # Pattern for source codes in window functions and partitions
            r'(?i)(?:PARTITION\s+BY|ORDER\s+BY)\s+(?:' + '|'.join(alias_list) + r')\.(?:data_srce_cde|ar_srce_cde)\s*=\s*[\'"]([^\'"]+)[\'"]',
            
            # Pattern for source codes in complex expressions
            r'(?i)(?:WHERE|AND|OR|ON|WHEN)\s+(?:' + '|'.join(alias_list) + r')\.(?:data_srce_cde|ar_srce_cde)\s*(?:=|IN|LIKE|REGEXP)\s*[\'"]([^\'"]+)[\'"]',
            
            # Pattern for source codes in subqueries
            r'(?i)(?:WHERE|AND|OR|ON|WHEN)\s+(?:' + '|'.join(alias_list) + r')\.(?:data_srce_cde|ar_srce_cde)\s+IN\s*\(\s*SELECT[\s\S]*?[\'"]([^\'"]+)[\'"]',
            
            # Pattern for source codes with quoted identifiers and special characters
            r'(?i)(?:WHERE|AND|OR|ON|WHEN)\s+["`]?(?:' + '|'.join(alias_list) + r')["`]?\.["`]?(?:data_srce_cde|ar_srce_cde)["`]?\s*=\s*[\'"]([^\'"]+)[\'"]',
            
            # Pattern for source codes in HAVING clauses
            r'(?i)HAVING\s+(?:' + '|'.join(alias_list) + r')\.(?:data_srce_cde|ar_srce_cde)\s*=\s*[\'"]([^\'"]+)[\'"]',
            
            # Pattern for source codes after newlines/whitespace
            r'(?i)(?:WHERE|AND|OR|ON|WHEN)\s*[\r\n]+\s*(?:' + '|'.join(alias_list) + r')\.(?:data_srce_cde|ar_srce_cde)\s*=\s*[\'"]([^\'"]+)[\'"]',
            
            # Pattern for source codes in complex JOIN conditions
            r'(?i)JOIN\s+[^\s]+\s+(?:AS\s+)?[^\s]+\s+ON\s+(?:' + '|'.join(alias_list) + r')\.(?:data_srce_cde|ar_srce_cde)\s*=\s*[\'"]([^\'"]+)[\'"]',
            
            # Pattern for source codes in EXISTS clauses
            r'(?i)EXISTS\s*\(\s*SELECT[\s\S]*?(?:' + '|'.join(alias_list) + r')\.(?:data_srce_cde|ar_srce_cde)\s*=\s*[\'"]([^\'"]+)[\'"]'
        ]
        
        def process_matches(pattern, query):
            matches = re.finditer(pattern, query)
            for match in matches:
                # Get all non-None groups (some patterns have multiple capture groups)
                values = [g for g in match.groups() if g is not None]
                for value in values:
                    print(f"Found source code: {value}")
                    source_codes.add(value)
        
        # Process original pattern first (preserved)
        process_matches(source_code_pattern, query)
        
        # Process additional patterns
        for pattern in additional_patterns:
            process_matches(pattern, query)
            
        # Also check for IN clauses with source codes (preserved)
        in_pattern = r'(?i)(?:WHERE|AND|OR|ON)\s+(?:' + '|'.join(alias_list) + r')\.(?:data_srce_cde|ar_srce_cde)\s+IN\s*\(([^)]+)\)'
        print(f"Searching for IN clauses with pattern: {in_pattern}")
        
        in_matches = re.finditer(in_pattern, query)
        for match in in_matches:
            in_values = match.group(1)
            print(f"Found IN clause values: {in_values}")
            # Extract individual values from the IN clause
            values = re.findall(r'[\'"]([^\'"]+)[\'"]', in_values)
            for value in values:
                print(f"Found source code in IN clause: {value}")
                source_codes.add(value)
                
        print(f"Final source codes for {table_name}: {source_codes}")
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
        },
        {
            "name": "CTE with Column List Test",
            "sql": """
                WITH cte1(col1, data_srce_cde) AS (
                    SELECT t1.col1, t1.data_srce_cde
                    FROM table1 t1
                    WHERE t1.data_srce_cde = 'A1'
                )
                SELECT c1.*
                FROM cte1 c1
            """
        },
        {
            "name": "Schema Qualified Tables Test",
            "sql": """
                SELECT t1.col1, t1.data_srce_cde
                FROM schema1.table1 t1
                JOIN schema2.table2 t2 ON t1.col1 = t2.col1
                WHERE t1.data_srce_cde = 'A1'
            """
        },
        {
            "name": "Quoted Identifiers Test",
            "sql": """
                SELECT "t1"."col1", "t1"."data_srce_cde"
                FROM "table1" "t1"
                JOIN `table2` `t2` ON "t1"."col1" = `t2`.`col1`
                WHERE "t1"."data_srce_cde" = 'A1'
            """
        },
        {
            "name": "Complex Column Expressions Test",
            "sql": """
                SELECT 
                    t1.col1 + t2.col1 as sum_col,
                    CASE WHEN t1.data_srce_cde = 'A1' THEN t2.ar_srce_cde ELSE t3.data_srce_cde END as derived_col,
                    ROW_NUMBER() OVER (PARTITION BY t1.data_srce_cde ORDER BY t1.col1) as rn
                FROM table1 t1
                JOIN table2 t2 ON t1.col1 = t2.col1
                JOIN table3 t3 ON t2.col1 = t3.col1
                WHERE t1.data_srce_cde = 'A1'
            """
        },
        {
            "name": "Special Characters Test",
            "sql": """
                SELECT t1."col-1", t1."data_srce_cde"
                FROM "table-1" t1
                WHERE t1."data_srce_cde" = 'A-1'
            """
        },
        {
            "name": "Complex Source Code Conditions Test",
            "sql": """
                SELECT t1.col1, t1.data_srce_cde
                FROM table1 t1
                WHERE (t1.data_srce_cde = 'A1' OR t1.data_srce_cde = 'B1')
                AND t1.ar_srce_cde IN (
                    SELECT t2.ar_srce_cde
                    FROM table2 t2
                    WHERE t2.data_srce_cde = 'C1'
                )
            """
        },
        {
            "name": "Window Functions Test",
            "sql": """
                SELECT 
                    t1.col1,
                    t1.data_srce_cde,
                    ROW_NUMBER() OVER (PARTITION BY t1.data_srce_cde ORDER BY t1.col1) as rn,
                    LAG(t1.data_srce_cde) OVER (ORDER BY t1.col1) as prev_source
                FROM table1 t1
                WHERE t1.data_srce_cde = 'A1'
            """
        },
        {
            "name": "Aggregate Functions Test",
            "sql": """
                SELECT 
                    t1.data_srce_cde,
                    COUNT(t1.col1) as count_col,
                    MAX(t2.ar_srce_cde) as max_source
                FROM table1 t1
                JOIN table2 t2 ON t1.col1 = t2.col1
                WHERE t1.data_srce_cde = 'A1'
                GROUP BY t1.data_srce_cde
            """
        },
        {
            "name": "Complex Subqueries Test",
            "sql": """
                SELECT t1.col1, t1.data_srce_cde
                FROM table1 t1
                WHERE t1.data_srce_cde IN (
                    SELECT t2.data_srce_cde
                    FROM table2 t2
                    WHERE t2.ar_srce_cde IN (
                        SELECT t3.ar_srce_cde
                        FROM table3 t3
                        WHERE t3.data_srce_cde = 'B1'
                    )
                )
            """
        },
        {
            "name": "Recursive CTE with Complex Conditions",
            "sql": """
                WITH RECURSIVE cte1 AS (
                    SELECT t1.col1, t1.data_srce_cde
                    FROM table1 t1
                    WHERE t1.data_srce_cde = 'A1'
                    UNION ALL
                    SELECT t1.col1, t1.data_srce_cde
                    FROM table1 t1
                    JOIN cte1 c ON t1.col1 = c.col1
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
            "name": "CTE with Column List",
            "sql": """
                WITH cte1(col1, data_srce_cde) AS (
                    SELECT t1.col1, t1.data_srce_cde
                    FROM table1 t1
                    WHERE t1.data_srce_cde = 'A1'
                )
                SELECT c1.*
                FROM cte1 c1
            """
        },
        {
            "name": "Schema Qualified Tables",
            "sql": """
                SELECT t1.col1, t1.data_srce_cde
                FROM schema1.table1 t1
                JOIN schema2.table2 t2 ON t1.col1 = t2.col1
                WHERE t1.data_srce_cde = 'A1'
            """
        },
        {
            "name": "Quoted Identifiers",
            "sql": """
                SELECT "t1"."col1", "t1"."data_srce_cde"
                FROM "table1" "t1"
                JOIN `table2` `t2` ON "t1"."col1" = `t2`.`col1`
                WHERE "t1"."data_srce_cde" = 'A1'
            """
        },
        {
            "name": "Complex Column Expressions",
            "sql": """
                SELECT 
                    t1.col1 + t2.col1 as sum_col,
                    CASE WHEN t1.data_srce_cde = 'A1' THEN t2.ar_srce_cde ELSE t3.data_srce_cde END as derived_col,
                    ROW_NUMBER() OVER (PARTITION BY t1.data_srce_cde ORDER BY t1.col1) as rn
                FROM table1 t1
                JOIN table2 t2 ON t1.col1 = t2.col1
                JOIN table3 t3 ON t2.col1 = t3.col1
                WHERE t1.data_srce_cde = 'A1'
            """
        },
        {
            "name": "Special Characters",
            "sql": """
                SELECT t1."col-1", t1."data_srce_cde"
                FROM "table-1" t1
                WHERE t1."data_srce_cde" = 'A-1'
            """
        },
        {
            "name": "Complex Source Code Conditions",
            "sql": """
                SELECT t1.col1, t1.data_srce_cde
                FROM table1 t1
                WHERE (t1.data_srce_cde = 'A1' OR t1.data_srce_cde = 'B1')
                AND t1.ar_srce_cde IN (
                    SELECT t2.ar_srce_cde
                    FROM table2 t2
                    WHERE t2.data_srce_cde = 'C1'
                )
            """
        },
        {
            "name": "Window Functions",
            "sql": """
                SELECT 
                    t1.col1,
                    t1.data_srce_cde,
                    ROW_NUMBER() OVER (PARTITION BY t1.data_srce_cde ORDER BY t1.col1) as rn,
                    LAG(t1.data_srce_cde) OVER (ORDER BY t1.col1) as prev_source
                FROM table1 t1
                WHERE t1.data_srce_cde = 'A1'
            """
        },
        {
            "name": "Aggregate Functions",
            "sql": """
                SELECT 
                    t1.data_srce_cde,
                    COUNT(t1.col1) as count_col,
                    MAX(t2.ar_srce_cde) as max_source
                FROM table1 t1
                JOIN table2 t2 ON t1.col1 = t2.col1
                WHERE t1.data_srce_cde = 'A1'
                GROUP BY t1.data_srce_cde
            """
        },
        {
            "name": "Complex Subqueries",
            "sql": """
                SELECT t1.col1, t1.data_srce_cde
                FROM table1 t1
                WHERE t1.data_srce_cde IN (
                    SELECT t2.data_srce_cde
                    FROM table2 t2
                    WHERE t2.ar_srce_cde IN (
                        SELECT t3.ar_srce_cde
                        FROM table3 t3
                        WHERE t3.data_srce_cde = 'B1'
                    )
                )
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