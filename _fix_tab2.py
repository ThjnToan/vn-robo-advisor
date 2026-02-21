"""One-shot fix script: restructures the broken tab2 download block in app.py"""

content = open("app.py", "r", encoding="utf-8").read()

# Target: the corrupted block where the download logic sits inside col_hdr_2a
# We replace with correctly indented block inside col_hdr_2b
old_marker = '    with col_hdr_2a:\n        st.subheader("Historical Tracking")\n    \n        # Use a JS Blob in a sandboxed iframe'
new_block = '''    with col_hdr_2a:
        st.subheader("Historical Tracking")
    
    with col_hdr_2b:
        pdf_holdings = {k: {"shares": v, "value": display_holding_values.get(k, 0)} for k, v in holdings.items() if v > 0}
        pdf_stream = export_tearsheet(display_nav, total_roi, json.dumps(pdf_holdings), cash)
        # Use a JS Blob in a sandboxed iframe'''

if old_marker in content:
    content = content.replace(old_marker, new_block, 1)
    open("app.py", "w", encoding="utf-8").write(content)
    print("SUCCESS: tab2 block fixed")
else:
    # Print what's around that area to help debug
    idx = content.find("with col_hdr_2a")
    print(f"NOT FOUND. Found 'with col_hdr_2a' at idx: {idx}")
    print(repr(content[idx:idx+300]))
