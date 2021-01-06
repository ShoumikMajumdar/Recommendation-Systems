# Recommendation-Systems

Extract the Credits.csv, Keywords.csv and movies_metedata.csv in Content_Webapp folder.

To run: <br>
cd to the Content_Webapp folder<br>
export FLASK_APP=app<br>
flask run


To query movies based on plot:
curl http://127.0.0.1:5000/plot?movie=Toy+Story

To query movies based on crew:
curl http://127.0.0.1:5000/crew?movie=Toy+Story

To query movies based on item-item collaborative filtering:
curl http://127.0.0.1:5000/item?movie=Toy+Story

