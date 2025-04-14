import pandas as pd
# Use sqlglot for robust parsing
import sqlglot
import sqlglot.expressions as exp
from typing import Dict, Set, List, Any, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
# Set sqlglot logger level higher to avoid excessive parsing logs unless debugging
logging.getLogger("sqlglot").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class SQLMetadataExtractor:
    def __init__(self, dialect: Optional[str] = None):
        """
        Initializes the extractor.

        Args:
            dialect (Optional[str]): The SQL dialect to use for parsing (e.g., 'snowflake', 'bigquery', 'postgres').
                                     If None, sqlglot will try to guess.
        """
        self.source_code_columns: Set[str] = {'AR_SRCE_CDE', 'DATA_SRCE_CDE'} # Use set for faster lookups
        self.dialect = dialect
        # No class-level state needed for results; process each query independently.

        # Updated _extract_sql_metadata method
    def _extract_sql_metadata(self, sql_str: str) -> Dict[str, Dict[str, Set[str]]]:
        """
        Extract metadata from SQL query using sqlglot AST traversal.
    
        Returns a dictionary mapping actual table names (lowercase, potentially schema-qualified
        if schema is used consistently) to their extracted columns and source codes.
        """
        if not sql_str or not isinstance(sql_str, str):
            logger.warning("Received empty or non-string SQL input.")
            return {}
    
        try:
            # Parse the SQL into an AST
            try:
                parsed_expressions = sqlglot.parse(sql_str, read=self.dialect)
                if not parsed_expressions:
                    logger.warning("SQL string parsed into an empty expression list.")
                    return {}
                parsed = parsed_expressions[0]
            except Exception as parse_error:
                logger.error(f"sqlglot failed to parse SQL: {parse_error}\nQuery snippet: {sql_str[:500]}...")
                return {}
    
            # Data Structures
            alias_to_table_map: Dict[str, str] = {}
            table_details: Dict[str, Dict[str, Set[str]]] = {}
            cte_names: Set[str] = set()
    
            # --- 1. Find CTEs ---
            if isinstance(parsed, exp.Query) and parsed.ctes:
                for cte in parsed.ctes:
                    if isinstance(cte, exp.CTE) and cte.alias:
                        cte_names.add(cte.alias.lower())
                        # Process CTE recursively
                        cte_metadata = self._extract_sql_metadata(cte.this.sql())
                        table_details[cte.alias.lower()] = cte_metadata.get(cte.alias.lower(), {"columns": set(), "source_codes": set()})
    
            logging.debug(f"Found CTE names: {cte_names}")
    
            # --- 2. Find Tables and Aliases (excluding CTEs) ---
            for table_expr in parsed.find_all(exp.Table):
                table_name_parts = []
                if table_expr.db:
                    table_name_parts.append(table_expr.db.lower())
                table_name_parts.append(table_expr.name.lower())
                full_table_name = ".".join(table_name_parts)
    
                if full_table_name in cte_names:
                    if table_expr.alias:
                        alias_to_table_map[table_expr.alias.lower()] = full_table_name
                    continue
    
                alias = table_expr.alias.lower() if table_expr.alias else full_table_name
                if full_table_name not in table_details:
                    table_details[full_table_name] = {'columns': set(), 'source_codes': set()}
                alias_to_table_map[alias] = full_table_name
                if alias != full_table_name:
                    alias_to_table_map[full_table_name] = full_table_name
    
            logging.debug(f"Found non-CTE tables: {list(table_details.keys())}")
            logging.debug(f"Alias map: {alias_to_table_map}")
    
            # --- 3. Find Columns ---
            processed_col_keys: Set[Tuple[str, str]] = set()
            for col_expr in parsed.find_all(exp.Column):
                table_ref = col_expr.table.lower() if col_expr.table else None
                column_name_upper = col_expr.name.upper()
    
                if not table_ref:
                    if len(alias_to_table_map) == 1:
                        table_ref = list(alias_to_table_map.keys())[0]
                    else:
                        continue
    
                actual_table = alias_to_table_map.get(table_ref)
                if actual_table and actual_table in table_details:
                    col_key = (actual_table, column_name_upper)
                    if col_key not in processed_col_keys:
                        table_details[actual_table]['columns'].add(column_name_upper)
                        processed_col_keys.add(col_key)
                        logging.debug(f"Found column: {actual_table}.{column_name_upper}")
    
            # --- 4. Find Source Codes in WHERE Clauses ---
            for where_clause in parsed.find_all(exp.Where):
                for condition in where_clause.find_all((exp.EQ, exp.In)):
                    column_ref_node = None
                    value_nodes = []
    
                    if isinstance(condition.left, exp.Column) and condition.left.table:
                        column_ref_node = condition.left
                        if isinstance(condition, exp.EQ):
                            if isinstance(condition.right, (exp.Literal, exp.String)):
                                value_nodes.append(condition.right)
                        elif isinstance(condition, exp.In) and isinstance(condition.right, (exp.Tuple, exp.List)):
                            value_nodes.extend(condition.right.expressions)
    
                    elif isinstance(condition.right, exp.Column) and condition.right.table:
                        if isinstance(condition, exp.EQ) and isinstance(condition.left, (exp.Literal, exp.String)):
                            column_ref_node = condition.right
                            value_nodes.append(condition.left)
    
                    if not column_ref_node:
                        continue
    
                    col_name_upper = column_ref_node.name.upper()
                    if col_name_upper in self.source_code_columns:
                        table_ref = column_ref_node.table.lower()
                        actual_table = alias_to_table_map.get(table_ref)
    
                        if actual_table and actual_table in table_details:
                            for val_node in value_nodes:
                                if isinstance(val_node, (exp.String, exp.Literal)):
                                    extracted_value = str(val_node.literal_value)
                                    table_details[actual_table]['source_codes'].add(extracted_value)
                                    logging.debug(f"Found source code: {actual_table}.{col_name_upper} -> {extracted_value}")
    
            # --- 5. Propagate CTE Metadata ---
            for cte_name in cte_names:
                if cte_name in alias_to_table_map:
                    actual_table = alias_to_table_map[cte_name]
                    if actual_table in table_details:
                        table_details[actual_table]['columns'].update(table_details[cte_name]['columns'])
                        table_details[actual_table]['source_codes'].update(table_details[cte_name]['source_codes'])
    
            # --- 6. Final Formatting ---
            final_result = {
                table: data for table, data in table_details.items()
                if data['columns'] or data['source_codes']
            }
            for table_data in final_result.values():
                table_data['columns'] = sorted(list(table_data['columns']))
                table_data['source_codes'] = sorted(list(table_data['source_codes']))
    
            logging.debug(f"Final extracted metadata: {final_result}")
            return final_result
    
        except Exception as e:
            logger.error(f"Unexpected error processing SQL query: {e}\nQuery: {sql_str[:500]}...", exc_info=True)
            return {}

    def _update_master_dict(self, master_dict: Dict[str, Dict[str, Set[str]]], current_metadata: Dict[str, Dict[str, List[str]]]):
        """Update master dictionary with new metadata. Expects lists from extraction, converts to sets."""
        for table, metadata in current_metadata.items():
            if table not in master_dict:
                # Initialize with sets
                master_dict[table] = {'columns': set(), 'source_codes': set()}
            # Update sets directly
            master_dict[table]['columns'].update(metadata.get('columns', [])) # Use .get for safety
            master_dict[table]['source_codes'].update(metadata.get('source_codes', []))


    def process_dataframe(self, df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
        """Process all queries from a DataFrame."""
        if 'query_text' not in df.columns:
            raise ValueError("DataFrame must contain 'query_text' column")

        # Use sets internally for efficiency, convert to lists at the end
        master_dict_sets: Dict[str, Dict[str, Set[str]]] = {}

        total_rows = len(df)
        for index, row in df.iterrows():
            query = row['query_text']
            if pd.isna(query) or not isinstance(query, str) or not query.strip():
                logger.warning(f"Skipping empty or invalid query at index {index} (Row {index + 1}/{total_rows})")
                continue

            logger.info(f"Processing query at index {index} (Row {index + 1}/{total_rows})...")
            # _extract_sql_metadata now returns dict with lists
            current_metadata = self._extract_sql_metadata(query)
            if current_metadata:
                 # Update the master dict which uses sets
                self._update_master_dict(master_dict_sets, current_metadata)
                logger.info(f"Successfully processed query at index {index} (Row {index + 1}/{total_rows})")
            else:
                 logger.warning(f"No metadata extracted for query at index {index} (Row {index + 1}/{total_rows})")


        # Convert final master dict sets to sorted lists
        final_master_dict: Dict[str, Dict[str, List[str]]] = {}
        for table, data in master_dict_sets.items():
            final_master_dict[table] = {
                'columns': sorted(list(data['columns'])),
                'source_codes': sorted(list(data['source_codes']))
            }
        logger.info(f"Finished processing DataFrame. Found metadata for {len(final_master_dict)} tables.")
        return final_master_dict

# --- Test Function ---
def test_queries():
    """Test the SQL metadata extraction with various cases."""
    # You can specify a dialect if most queries conform to it
    # extractor = SQLMetadataExtractor(dialect='snowflake')
    extractor = SQLMetadataExtractor() # Let sqlglot guess

    test_cases = [
        # --- Basic Cases ---
        {
            "name": "Simple Select",
            "sql": "SELECT t1.colA, t1.DATA_SRCE_CDE FROM db.schema.table1 t1 WHERE t1.DATA_SRCE_CDE = 'ABC'"
        },
        {
            "name": "Simple Join",
            "sql": """
                SELECT t1.colA, t2.colB, t1.DATA_SRCE_CDE, t2.AR_SRCE_CDE
                FROM table1 t1
                JOIN table2 t2 ON t1.id = t2.fk_id
                WHERE t1.DATA_SRCE_CDE IN ('A1', 'B2') AND t2.AR_SRCE_CDE = 'X1'
            """
        },
         {
            "name": "Schema Qualification",
            "sql": """
                SELECT sch1.tbl1.colA, sch2.tbl2.colB, sch1.tbl1.DATA_SRCE_CDE
                FROM schema1.table1 sch1.tbl1 -- Alias includes schema part (check sqlglot behavior)
                JOIN schema2.table2 AS sch2.tbl2 ON sch1.tbl1.id = sch2.tbl2.id
                WHERE sch1.tbl1.DATA_SRCE_CDE = 'S1'
            """ # Note: sqlglot might parse 'sch1.tbl1' differently depending on dialect. Testing needed.
              # Standard alias is usually just 'tbl1'. Let's adjust for clarity:
        },
        {
            "name": "Schema Qualification Standard Alias",
            "sql": """
                SELECT t1.colA, t2.colB, t1.DATA_SRCE_CDE
                FROM schema1.table1 AS t1
                JOIN schema2.table2 t2 ON t1.id = t2.id
                WHERE t1.DATA_SRCE_CDE = 'S1' AND t2.AR_SRCE_CDE IN ('S2', 'S3')
            """
        },
        # --- CTE Cases (Copied from previous approach) ---
         {
            "name": "Basic CTE Test",
            "sql": """
                WITH cte1 AS (
                    SELECT t1.col1, t1.data_srce_cde
                    FROM table1 t1
                    WHERE t1.data_srce_cde IN ('A1', 'B2')
                )
                SELECT c1.col1, c1.data_srce_cde -- Referencing CTE column
                FROM cte1 c1
                JOIN real_table rt ON c1.col1 = rt.id -- Join CTE with real table
                WHERE rt.some_col = 123 -- Condition on real table
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
                SELECT c1.col1, c2.col1, c1.data_srce_cde, c2.ar_srce_cde
                FROM cte1 c1
                JOIN cte2 c2 ON c1.col1 = c2.col1
                JOIN table3 t3 ON c2.col1 = t3.id -- Join CTE with real table
                WHERE t3.AR_SRCE_CDE = 'Y1' -- Condition on real table
            """
        },
        {
            "name": "Recursive CTE Test",
            "sql": """
                WITH RECURSIVE number_cte (n) AS (
                    SELECT 1
                    UNION ALL
                    SELECT n + 1 FROM number_cte WHERE n < 5
                )
                SELECT n.n, t1.colA, t1.DATA_SRCE_CDE
                FROM number_cte n
                JOIN table1 t1 ON n.n = t1.id
                WHERE t1.DATA_SRCE_CDE = 'REC'
            """
        },
        {
            "name": "Nested CTEs Test (Non-Standard but Parsable)",
            # sqlglot *can* often parse this, treating cte2 as defined within cte1's scope
            # but the extraction logic should correctly identify table1 as the base table.
            "sql": """
                WITH cte1 AS (
                    WITH cte2 AS (
                        SELECT t1.col1, t1.data_srce_cde
                        FROM table1 t1
                        WHERE t1.data_srce_cde = 'A1'
                    )
                    SELECT c2.col1, c2.data_srce_cde
                    FROM cte2 c2
                )
                SELECT c1.col1, c1.data_srce_cde
                FROM cte1 c1
            """
        },
        {
            "name": "CTE with Subqueries Test",
            "sql": """
                WITH cte1 AS (
                    SELECT t1.col1, t1.data_srce_cde
                    FROM table1 t1
                    WHERE t1.data_srce_cde IN (
                        SELECT t2.data_srce_cde -- This subquery uses table2
                        FROM table2 t2
                        WHERE t2.ar_srce_cde = 'X1'
                    )
                )
                SELECT c1.col1, c1.data_srce_cde
                FROM cte1 c1
                LEFT JOIN table3 t3 ON c1.col1 = t3.id
                WHERE t3.DATA_SRCE_CDE = 'Z1'
            """
        },
        {
            "name": "Mega CTE Test with Real Table Join",
            "sql": """
WITH cte1 AS (SELECT t1.col1, t1.data_srce_cde FROM table1 t1 WHERE t1.data_srce_cde IN ('A1', 'B2')),
cte2 AS (SELECT t2.col1, t2.ar_srce_cde FROM table2 t2 WHERE t2.ar_srce_cde = 'X1'),
cte3 AS (SELECT t3.col1, t3.data_srce_cde FROM table3 t3 WHERE t3.data_srce_cde = 'C3'),
cte4 AS (SELECT t4.col1, t4.ar_srce_cde FROM table4 t4 WHERE t4.ar_srce_cde = 'Y2'),
cte5 AS ( -- This CTE uses table5
    WITH nested_cte AS (SELECT t5.col1, t5.data_srce_cde FROM table5 t5 WHERE t5.data_srce_cde = 'D4')
    SELECT n.col1, n.data_srce_cde FROM nested_cte n
),
cte6 AS (SELECT t6.col1, t6.ar_srce_cde FROM table6 t6 WHERE t6.ar_srce_cde IN ('E5','F6')),
cte7 AS (SELECT t7.col1, t7.data_srce_cde FROM table7 t7 WHERE t7.data_srce_cde = 'G7'),
cte8 AS (SELECT t8.col1, t8.ar_srce_cde FROM table8 t8 WHERE t8.ar_srce_cde = 'H8')
SELECT
    c1.col1, c2.col1, c3.col1, c4.col1, c5.col1, c6.col1, c7.col1, c8.col1, -- CTE columns
    rt.col_real, rt.ar_srce_cde -- Real table columns
FROM cte1 c1
JOIN cte2 c2 ON c1.col1 = c2.col1
JOIN cte3 c3 ON c2.col1 = c3.col1
JOIN cte4 c4 ON c3.col1 = c4.col1
JOIN cte5 c5 ON c4.col1 = c5.col1
JOIN cte6 c6 ON c5.col1 = c6.col1
JOIN cte7 c7 ON c6.col1 = c7.col1
JOIN cte8 c8 ON c7.col1 = c8.col1
LEFT JOIN real_table rt ON c8.col1 = rt.id -- Join on a real table
WHERE rt.ar_srce_cde = 'Z9' AND c1.data_srce_cde = 'A1' -- Conditions on CTE and real table
            """
        },
        # --- Edge Cases ---
        {
            "name": "Query with only real tables",
            "sql": """
                SELECT t1.col1, t2.col2, t1.data_srce_cde, t2.ar_srce_cde
                FROM real_table1 t1
                JOIN real_table2 t2 ON t1.id = t2.fk_id
                WHERE t1.data_srce_cde = 'R1' AND t2.ar_srce_cde IN ('R2', 'R3')
            """
        },
        {
            "name": "Source Code in different conditions",
            "sql": """
                SELECT t1.colA
                FROM table1 t1
                WHERE (t1.DATA_SRCE_CDE = 'C1' OR t1.DATA_SRCE_CDE = 'C2')
                  AND t1.AR_SRCE_CDE IN ('C3', 'C4')
            """
        },
        {
            "name": "Column names matching source code column names",
            "sql": """
                SELECT t1.DATA_SRCE_CDE as source_code_alias, t1.colA, t2.AR_SRCE_CDE
                FROM table1 t1 JOIN table2 t2 ON t1.id = t2.id
                WHERE t1.DATA_SRCE_CDE = 'V1' AND t2.AR_SRCE_CDE = 'V2'
            """
        },
        {
            "name": "No relevant columns or source codes",
            "sql": "SELECT t1.colA, t2.colB FROM table1 t1 JOIN table2 t2 ON t1.id=t2.id WHERE t1.other_col = 5"
        },
        {
            "name": "Source code column on right side of EQ",
             "sql": "SELECT t1.colA FROM table1 t1 WHERE 'RHS' = t1.DATA_SRCE_CDE"
        }

    ]

    master_results_sets: Dict[str, Dict[str, Set[str]]] = {}

    for i, test in enumerate(test_cases):
        print(f"\n=== Running Test {i+1}: {test['name']} ===")
        # print("\nSQL Query:")
        # print(test['sql'])
        try:
            # Call the extraction method directly for isolated test result
            result = extractor._extract_sql_metadata(test['sql'])
            print("\nExtracted Metadata (Isolated):")
            if not result:
                print("No metadata extracted.")
            # Convert to lists for printing consistency with final output
            printable_result = {}
            for table, metadata in result.items():
                 printable_result[table] = {
                     'columns': sorted(list(metadata.get('columns', set()))),
                     'source_codes': sorted(list(metadata.get('source_codes', set())))
                 }

            for table, metadata in printable_result.items():
                print(f"\nTable: {table}")
                print(f"  Columns: {metadata['columns']}")
                print(f"  Source Codes: {metadata['source_codes']}")

            # Simulate accumulation
            extractor._update_master_dict(master_results_sets, result)

        except Exception as e:
            print(f"\nError processing test case '{test['name']}': {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Print final accumulated results (converted to lists)
    print("\n\n=== Final Accumulated Results ===")
    final_master_dict: Dict[str, Dict[str, List[str]]] = {}
    for table, data in master_results_sets.items():
        final_master_dict[table] = {
            'columns': sorted(list(data['columns'])),
            'source_codes': sorted(list(data['source_codes']))
        }

    for table, metadata in final_master_dict.items():
        print(f"\nTable: {table}")
        print(f"  Columns: {metadata['columns']}")
        print(f"  Source Codes: {metadata['source_codes']}")


if __name__ == "__main__":
    test_queries()
