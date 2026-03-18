// ChisCode — MongoDB Initialization
// This script runs once when the MongoDB container first starts.
// Creates the chiscode database and sets up the application user.

db = db.getSiblingDB('chiscode');

db.createCollection('users');
db.createCollection('projects');
db.createCollection('project_versions');
db.createCollection('sessions');
db.createCollection('templates');

print('ChisCode MongoDB initialized successfully.');
