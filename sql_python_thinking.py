# How Python interacts with database
class Database:
  def connect(self):
    return "Database connected"
  def fetch_data(self):
    return ["Student1","Student2","Student3"]

db = Database()
print(db.connect())
print(db.fetch_data())
