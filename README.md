The following are the steps to run the project of Face Mask Detection and Face Mask Removal using GAN
1) Go to the environment where our working directory of the project exists. Here we used environmental folder as env.
2) Go to the scripts in env folder.
3) Open the command prompt in this relative directory.
4) Now use the command activate to activate the created environment to run our django project.
5) In the command prompt now go back to the previous folder which is called as env by using the command "cd .."
6) Now go to the project working directory by using the commands " cd digidress " and " cd FaceMask"
7) Now run the command "python manage.py runserver" this gives the URL of the project running localhost server. Commonly as http://127.0.0.1:8000/
8) for migrations the command is "python manage.py migrate"
9) To create the super user in django the command is "python manage.py createsuperuser"
10) To install modules in text documents on django the command is "pip install -r requirements.txt"


The following are the steps or instructions to use our project
1) The first step will be the login page. You can log in if you already have account in Face Mask website platform otherwise you can create your account by just tapping on the registration button and filling the form.
2) After successful login you'll be redirected to the Face Mask tool Page where you can see our "Face Mask Detection" and "Face Mask Removal" buttons.
3) When you click on Face Mask Detection button it will navigate to another web page.On that web page consists of two buttons first button is for face mask detection on Images and second button is face mask detection on Live video streaming.
4) when you click on Face Mask Removal button it will open face mask removal website it will work on images only.


The following are the instructions that should consider to upload the user image in the Face Mask Detection and Face Mask Removal:
1) To recognize the Images in face mask detection you should upload quality image in the recognize folder.
2) To Face Mask Removal the upload image should only face image and doesn't consists of our body image on that and Image should be consists of Lighting and high quality image
3) There should be no blur in the image.

output Images:
