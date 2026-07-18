from app.vectorstore import query_candidates

results = query_candidates("casual dinner", category="top")
for r in results:
    print(r)
