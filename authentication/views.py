from django.core.mail import EmailMessage
from django.shortcuts import redirect, render
from django.contrib.sites.shortcuts import get_current_site
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate,login
from facemask import settings
from django.core.mail import send_mail
from django.db import IntegrityError
from django.template.loader import render_to_string
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_encode
from . tokens import generate_token
from django.shortcuts import render
from facemask.Maskpic import maskdetect,maskdetect1
from facemask.Removal import MaskRemoval
from django.http.response import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from .forms import ImageUploadForm
from django.core.files.base import ContentFile
from tempfile import NamedTemporaryFile
from .models import YourModel 
from django.http import JsonResponse
import os



def gen(camera):
    while True:
        frame =camera.get_frame()
        yield (b'--frame\r\n' b'Content-Type:image/jpeg\r\n\r\n'+ frame +b'\r\n\r\n')

def gen_input(input_image):
    mask_detector=maskdetect1()
    frame = mask_detector.get_frame1(input_image)
    return frame

def clear_temporary_files(request):
    temp_dir = os.path.join(settings.MEDIA_ROOT)
    for file_name in os.listdir(temp_dir):
        YourModel.objects.filter(image=file_name).delete()
        file_path = os.path.join(temp_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    # return JsonResponse({'message': 'Temporary files cleared successfully'})
    return redirect('/Removal')
# def clear_temporary_files(request):
#     if request.method == 'POST':
#         temporary_files = request.POST.getlist('temporary_files[]', [])
#         temp_dir = os.path.join(settings.MEDIA_ROOT)
#         for file_name in temporary_files:
#             file_path = os.path.join(temp_dir, file_name)
#             if os.path.exists(file_path):
#                 os.remove(file_path)
#         return JsonResponse({'message': 'Temporary files cleared successfully'})
#     return JsonResponse({'error': 'Invalid request method'})




# Create your views here.
def Main(request):
    return render(request,"authentication/Main.html")

def Register(request):
    if request.method == 'POST':
        Name = request.POST['name']
        Username = request.POST['username']
        Email = request.POST['email']
        Password = request.POST['password']
        Confirmpassword = request.POST['confirmpassword']

        if User.objects.filter(username=Username).exists():
            messages.error(request, "UserName already exists...! Please try some other user name ...")
        elif User.objects.filter(email=Email).exists():
            messages.error(request, "Email already registered...!")
        elif len(Username) > 10:
            messages.error(request, "Username must be under 10 characters...!")
        elif Password != Confirmpassword:
            messages.error(request, "Password and confirm password must be the same...!")
        elif Username.isalnum() == False:
            messages.error(request, "Username must be alphanumeric...!")
        else:
            try:
                myuser = User.objects.create_user(username=Username, email=Email, password=Password)
                myuser.Name = Name
                myuser.is_active = True
                myuser.save()
                messages.success(request,"Hiii "+ myuser.Name +" Your account has been created successfully in face mask website and we had sended the confirmation mail in order to activate your account !  ")

                #welcome Email
                subject="Welcome to Face Mask Tool website...!"
                message="Hello "+ myuser.Name  +" !! \n"+" Welcome to Face Mask Tool website...! \n Thank You for visiting our website. \n We also sent to you a confirmation email ,Please confirm your email address in order to activate your account...\n\n Thanking You, \n Saran "
                from_email=settings.EMAIL_HOST_USER
                to_list=[myuser.email]
                send_mail(subject,message,from_email,to_list,fail_silently=True)

                #email address confirmation email
                current_site=get_current_site(request)
                context={
                'name':myuser.Name,
                'domain':current_site.domain,
                'uid':urlsafe_base64_encode(force_bytes(myuser.pk)),
                'token':generate_token.make_token(myuser)
                }
                message1=render_to_string('email_confirmation.html',context)
                email=EmailMessage("confirm your email ",
                    message1,
                    settings.EMAIL_HOST_USER,
                    [myuser.email]
                )
                email.fail_silently= True
                email.send()

            except IntegrityError:
                messages.error(request, "An error occurred while registering the user.")

    return render(request,"authentication/Register.html")
usernameicon=''
def Login(request):
    global usernameicon
    if request.method == 'POST':
        usname=request.POST['username']
        pass1=request.POST['password']
        user=authenticate(username=usname,password=pass1)
        usernameicon=usname
        if user is not None:
            login(request,user)
            return render(request,"authentication/Mask.html",{'fullname':usname})

        else:
            messages.error(request,".......... Bad Credentials ............")

    return render(request,"authentication/Login.html")


def Mask(request):
    global usernameicon
    return render(request,"authentication/Mask.html",{'fullname':usernameicon})

def Maskchoice(request):
    global usernameicon
    return render(request,"authentication/Maskchoice.html",{'fullname':usernameicon})


def Maskpic(request): 
    global usernameicon
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img = form.cleaned_data.get("image")
            predicted_image_data = gen_input(img)
            predicted_image = YourModel()
            predicted_image.image.save('predicted_image.jpg', ContentFile(predicted_image_data), save=False)
            predicted_image.save()
            return render(request, "authentication/Maskpic.html", {'predicted_image': predicted_image})
    else:
        form = ImageUploadForm()
    return render(request, "authentication/Maskpic.html", {'form': form,'fullname':usernameicon})


def video(request):
    return StreamingHttpResponse(gen(maskdetect()),content_type='multipart/x-mixed-replace; boundary=frame')



def Removal(request):
    global usernameicon
    predicted_image_urls = []

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():

            img = form.cleaned_data.get("image")
            mask_detector = MaskRemoval()
            predicted_images= mask_detector.upload_file(img)
            print(img)
            # if predicted_images is None:
            #      print("Error: Removal_input function returned None")
            #      predicted_images = []
            for idx, image_data in enumerate(predicted_images):
                temp_file_name = f"predicted_image_{idx}.jpg"
                temp_file_path = os.path.join(settings.MEDIA_ROOT, temp_file_name)
                image_data.save(temp_file_path)
                # Construct the URL path relative to MEDIA_URL
                temp_file_url = os.path.join(settings.MEDIA_URL, temp_file_name)
                predicted_image_urls.append(temp_file_url)
            return render(request, "authentication/Removal.html", {'predicted_image_urls': predicted_image_urls})
    else:
        form = ImageUploadForm()

    return render(request, "authentication/Removal.html", {'form':form,'fullname':usernameicon})



def activate(request,uid64,token):
    try:
        uid =force_str(urlsafe_base64_encode(uid64))
        myuser=User.objects.get(pk=uid)
    except(TypeError,ValueError,OverflowError,User.DoesNotExist):
        myuser =None

    if myuser is not None and generate_token.check_token(myuser,token):
        myuser.is_active=True
        myuser.save()
        login(request,myuser)
    else:
        return render(request,'activation_failed.html')
