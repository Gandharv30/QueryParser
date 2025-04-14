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
        self.alias_map = {}  # Track column aliases
        self.table_columns = {}  # Track actual columns per table

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
            self.alias_map = {}  # Reset alias map for new query
            self.table_columns = {}  # Reset table columns for new query
            
            # Get all tables and initialize result dictionary
            tables = parser.tables
            if not tables:
                logger.warning("No tables found in query")
                return {}
            
            # Initialize tables with case-insensitive handling
            for table in tables:
                clean_table = table.split('.')[-1].strip('`"[] ')
                result[clean_table] = [set(), set()]  # [columns, source_codes]
                self.table_columns[clean_table] = set()
            
            # Extract column aliases and table references
            self._extract_column_aliases(sql)
            self._extract_table_references(sql)
            
            # Process columns with improved alias awareness
            self._process_columns(parser.columns, result)
            
            # Extract and associate source codes per table
            table_source_codes = self._extract_table_source_codes(sql, tables)
            
            # Add source codes only to their respective tables
            for table, codes in table_source_codes.items():
                clean_table = table.split('.')[-1].strip('`"[] ')
                if clean_table in result:
                    result[clean_table][1].update(codes)
            
            # Clean up columns (remove IN from column lists)
            for table in result:
                result[table][0] = {col for col in result[table][0] if col.upper() != 'IN'}
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing SQL: {str(e)}")
            raise

    def _extract_column_aliases(self, sql: str):
        """Extract column aliases from the query."""
        # Match patterns like "column as alias" or "column alias"
        alias_pattern = r'(\w+(?:\.\w+)?)\s+(?:AS\s+)?(\w+)(?:\s*,|\s*FROM|\s*$)'
        matches = re.finditer(alias_pattern, sql, re.IGNORECASE)
        for match in matches:
            original, alias = match.groups()
            self.alias_map[original.upper()] = alias.upper()

    def _extract_table_references(self, sql: str):
        """Extract table references and their actual columns."""
        # Match table references in FROM and JOIN clauses
        table_pattern = r'(?:FROM|JOIN)\s+(?:\w+\.)?(\w+)(?:\s+(?:AS\s+)?(\w+))?'
        matches = re.finditer(table_pattern, sql, re.IGNORECASE)
        for match in matches:
            table, alias = match.groups()
            if alias:
                self.alias_map[alias.upper()] = table.upper()

    def _process_columns(self, columns: List[str], result: Dict[str, List[Set]]):
        """Process columns and add them to their respective tables."""
        processed_columns = {}  # Track processed columns to avoid duplicates
        
        for column in columns:
            if column == '*' or column.endswith('.*'):
                continue
            
            parts = column.split('.')
            if len(parts) > 1:
                # Handle qualified columns (table.column)
                col = parts[-1].strip('`"[] ')
                if col == '*':
                    continue
                    
                table_ref = parts[-2].strip('`"[] ').upper()
                table = self.alias_map.get(table_ref, table_ref)
                
                if table.lower() in result:
                    # Use aliased name if exists, otherwise use original
                    col_upper = col.upper()
                    final_col = self.alias_map.get(f"{table}.{col_upper}", 
                                                 self.alias_map.get(col_upper, col_upper))
                    
                    # Add to processed columns to avoid duplicates
                    key = f"{table}.{final_col}"
                    if key not in processed_columns:
                        result[table.lower()][0].add(final_col)
                        processed_columns[key] = True
                        self.table_columns[table.lower()].add(final_col)
            else:
                # Handle unqualified columns
                col = parts[0].strip('`"[] ')
                if col == '*':
                    continue
                
                col_upper = col.upper()
                final_col = self.alias_map.get(col_upper, col_upper)
                
                # Add to all tables if unqualified
                for table in result:
                    key = f"{table}.{final_col}"
                    if key not in processed_columns:
                        result[table][0].add(final_col)
                        processed_columns[key] = True
                        self.table_columns[table].add(final_col)

    def _extract_table_source_codes(self, sql: str, tables: List[str]) -> Dict[str, Set[str]]:
        """Extract source codes associated with specific tables."""
        table_source_codes = {}
        table_aliases = {}
        
        # First pass: collect table aliases
        for table in tables:
            clean_table = table.split('.')[-1].strip('`"[] ').lower()
            table_source_codes[clean_table] = set()
            
            # Find table aliases in the query
            alias_pattern = fr"(?:FROM|JOIN)\s+(?:\w+\.)?{clean_table}\s+(?:AS\s+)?(\w+)"
            alias_matches = re.finditer(alias_pattern, sql, re.IGNORECASE)
            for match in alias_matches:
                alias = match.group(1)
                table_aliases[alias.lower()] = clean_table
        
        # Add the original table names as their own aliases
        for table in tables:
            clean_table = table.split('.')[-1].strip('`"[] ').lower()
            table_aliases[clean_table] = clean_table
        
        # Second pass: extract source codes with proper table association
        for col in self.source_code_columns:
            # Pattern to match both direct table references and aliases
            pattern = fr'([a-zA-Z0-9_]+)\.{col}\s+IN\s*\(([\s\S]*?)\)'
            matches = re.finditer(pattern, sql, re.IGNORECASE)
            
            for match in matches:
                table_ref = match.group(1).lower()
                values_str = match.group(2)
                
                # Only process if we can map this reference to a real table
                if table_ref in table_aliases:
                    actual_table = table_aliases[table_ref]
                    
                    # Extract values from the IN clause
                    values = re.findall(r"'([^']*)'|\"([^\"]*)\"|(\w+)", values_str)
                    for value_tuple in values:
                        value = next((v for v in value_tuple if v), None)
                        if value:
                            table_source_codes[actual_table].add(value.strip())
        
        return table_source_codes

    def _update_master_dict(self, current_metadata: Dict[str, List[Set]]):
        """Update master dictionary with new metadata."""
        for table, (columns, source_codes) in current_metadata.items():
            if table not in self.master_dict:
                self.master_dict[table] = [set(), set()]
            
            # Update columns (case-insensitive)
            current_columns = {col.upper() for col in columns}
            self.master_dict[table][0].update(current_columns)
            
            # Update source codes
            self.master_dict[table][1].update(source_codes)

def test_queries():
    """Run comprehensive test cases covering various SQL patterns."""
    
    test_cases = [
        {
            "name": "Basic SELECT with Multiple Source Codes",
            "query": """
            SELECT 
                t1.column_a as col_a,
                t2.column_b as col_b,
                t1.AR_SRCE_CDE,
                t2.DATA_SRCE_CDE
            FROM table1 t1
            JOIN table2 t2 ON t1.id = t2.id
            WHERE t1.AR_SRCE_CDE IN ('A1', 'B1')
            AND t2.DATA_SRCE_CDE IN ('X1', 'Y1');
            """
        },
        {
            "name": "Multiple CTEs with Nested Source Codes",
            "query": """
            WITH source_data AS (
                SELECT 
                    s.source_id,
                    s.AR_SRCE_CDE,
                    d.DATA_SRCE_CDE
                FROM sources s
                JOIN details d ON s.id = d.id
                WHERE s.AR_SRCE_CDE IN ('S1', 'S2')
                AND d.DATA_SRCE_CDE IN ('D1', 'D2')
            ),
            customer_data AS (
                SELECT 
                    c.customer_id,
                    c.AR_SRCE_CDE,
                    p.DATA_SRCE_CDE,
                    p.preference_type
                FROM customers c
                LEFT JOIN preferences p ON c.id = p.customer_id
                WHERE c.AR_SRCE_CDE IN ('C1', 'C2')
                AND p.DATA_SRCE_CDE IN ('P1', 'P2')
            ),
            final_data AS (
                SELECT 
                    sd.*,
                    cd.*,
                    CASE 
                        WHEN sd.AR_SRCE_CDE IN ('S1') THEN 'HIGH'
                        WHEN cd.DATA_SRCE_CDE IN ('P1') THEN 'MEDIUM'
                        ELSE 'LOW'
                    END as priority
                FROM source_data sd
                JOIN customer_data cd ON sd.source_id = cd.customer_id
            )
            SELECT * FROM final_data;
            """
        },
        {
            "name": "UNION with Different Source Codes",
            "query": """
            SELECT 
                t1.id,
                t1.AR_SRCE_CDE,
                'Type1' as source_type
            FROM table1 t1
            WHERE t1.AR_SRCE_CDE IN ('A1', 'A2')
            UNION ALL
            SELECT 
                t2.id,
                t2.DATA_SRCE_CDE,
                'Type2' as source_type
            FROM table2 t2
            WHERE t2.DATA_SRCE_CDE IN ('B1', 'B2')
            UNION ALL
            SELECT 
                t3.id,
                t3.AR_SRCE_CDE,
                'Type3' as source_type
            FROM table3 t3
            WHERE t3.AR_SRCE_CDE IN ('C1', 'C2');
            """
        },
        {
            "name": "Nested Subqueries with Source Codes",
            "query": """
            SELECT 
                m.member_id,
                m.AR_SRCE_CDE,
                (SELECT p.product_name 
                 FROM products p 
                 WHERE p.id = m.product_id
                 AND p.AR_SRCE_CDE IN ('P1', 'P2')
                ) as product,
                (SELECT s.segment_name
                 FROM segments s
                 WHERE s.id = m.segment_id
                 AND s.DATA_SRCE_CDE IN ('S1', 'S2')
                ) as segment
            FROM members m
            WHERE m.AR_SRCE_CDE IN ('M1', 'M2')
            AND EXISTS (
                SELECT 1 
                FROM validation v
                WHERE v.member_id = m.member_id
                AND v.AR_SRCE_CDE IN ('V1', 'V2')
            );
            """
        },
        {
            "name": "Complex Joins with Multiple Source Code Conditions",
            "query": """
            SELECT 
                t1.transaction_id,
                t1.AR_SRCE_CDE as trans_source,
                t2.DATA_SRCE_CDE as cust_source,
                t3.AR_SRCE_CDE as prod_source,
                t4.DATA_SRCE_CDE as loc_source
            FROM transactions t1
            LEFT JOIN customers t2 
                ON t1.cust_id = t2.id
                AND t2.DATA_SRCE_CDE IN ('C1', 'C2')
            INNER JOIN products t3
                ON t1.prod_id = t3.id
                AND t3.AR_SRCE_CDE IN ('P1', 'P2')
            RIGHT JOIN locations t4
                ON t1.loc_id = t4.id
                AND t4.DATA_SRCE_CDE IN ('L1', 'L2')
            WHERE t1.AR_SRCE_CDE IN ('T1', 'T2')
            AND EXISTS (
                SELECT 1 FROM rules r 
                WHERE r.transaction_id = t1.transaction_id
                AND r.AR_SRCE_CDE IN ('R1', 'R2')
            );
            """
        },
        {
            "name": "Window Functions with Source Codes",
            "query": """
            WITH ranked_data AS (
                SELECT 
                    t.transaction_id,
                    t.AR_SRCE_CDE,
                    t.amount,
                    ROW_NUMBER() OVER (
                        PARTITION BY t.AR_SRCE_CDE 
                        ORDER BY t.amount DESC
                    ) as rank
                FROM transactions t
                WHERE t.AR_SRCE_CDE IN ('A1', 'A2', 'A3')
            ),
            filtered_sources AS (
                SELECT 
                    s.source_id,
                    s.DATA_SRCE_CDE,
                    COUNT(*) OVER (
                        PARTITION BY s.DATA_SRCE_CDE
                    ) as source_count
                FROM sources s
                WHERE s.DATA_SRCE_CDE IN ('B1', 'B2', 'B3')
            )
            SELECT 
                rd.*,
                fs.source_count,
                CASE 
                    WHEN rd.AR_SRCE_CDE IN ('A1') AND fs.DATA_SRCE_CDE IN ('B1') THEN 'Priority'
                    ELSE 'Standard'
                END as category
            FROM ranked_data rd
            JOIN filtered_sources fs ON rd.transaction_id = fs.source_id
            WHERE rd.rank <= 10;
            """
        },
        {
            "name": "Complex Mega Query with Multiple Patterns",
            "query": """
            WITH revenue_sources AS (
                SELECT 
                    rev.transaction_id,
                    rev.AR_SRCE_CDE as primary_source,
                    src.DATA_SRCE_CDE as secondary_source,
                    rev.amount as revenue_amount,
                    src.cost_amount,
                    COALESCE(rev.adj_factor, src.adj_factor, 1.0) as adjustment,
                    ROW_NUMBER() OVER (
                        PARTITION BY rev.AR_SRCE_CDE 
                        ORDER BY rev.amount DESC
                    ) as revenue_rank
                FROM schema1.revenue_transactions rev
                LEFT JOIN schema2.source_details src 
                    ON rev.transaction_id = src.trans_id
                    AND rev.AR_SRCE_CDE IN ('A1', 'B1', 'C1')
                    AND src.DATA_SRCE_CDE IN ('X1', 'Y1', 'Z1')
            ),
            customer_profile AS (
                SELECT 
                    c.customer_id,
                    c.AR_SRCE_CDE as cust_source,
                    d.DATA_SRCE_CDE as demo_source,
                    d.region_code,
                    p.preference_code,
                    (SELECT MAX(h.score)
                     FROM history h
                     WHERE h.customer_id = c.customer_id
                     AND h.AR_SRCE_CDE IN ('H1', 'H2')
                    ) as max_score
                FROM schema3.customer_master c
                INNER JOIN schema4.demographics d 
                    ON c.customer_id = d.cust_id
                    AND c.AR_SRCE_CDE IN ('A2', 'B2', 'C2')
                    AND d.DATA_SRCE_CDE IN ('X2', 'Y2', 'Z2')
                LEFT JOIN schema5.preferences p
                    ON c.customer_id = p.cust_id
                    AND p.AR_SRCE_CDE IN ('P1', 'P2', 'Q1', 'Q2')
            )
            SELECT 
                rs.*,
                cp.*,
                CASE 
                    WHEN rs.primary_source IN (
                        SELECT src.AR_SRCE_CDE 
                        FROM schema9.source_master src
                        WHERE src.AR_SRCE_CDE IN ('G1', 'G2')
                    ) THEN 'PRIMARY'
                    WHEN rs.secondary_source IN (
                        SELECT src.DATA_SRCE_CDE
                        FROM schema9.source_master src
                        WHERE src.DATA_SRCE_CDE IN ('H1', 'H2')
                    ) THEN 'SECONDARY'
                    ELSE 'OTHER'
                END as source_type
            FROM revenue_sources rs
            LEFT JOIN customer_profile cp 
                ON rs.primary_source = cp.cust_source
            WHERE EXISTS (
                SELECT 1 
                FROM schema10.validation_rules vr
                WHERE vr.AR_SRCE_CDE IN ('V1', 'V2')
                AND vr.rule_type = 'ACTIVE'
            )
            UNION ALL
            SELECT 
                rs.*,
                cp.*,
                'HISTORICAL' as source_type
            FROM revenue_sources rs
            RIGHT JOIN customer_profile cp 
                ON rs.secondary_source = cp.demo_source
            WHERE rs.revenue_rank <= 5;
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