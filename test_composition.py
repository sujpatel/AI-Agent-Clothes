from app.composition import compose_outfit

result = compose_outfit(occasion="casual dinner", location="Chicago")
print(result.model_dump_json(indent=2))
