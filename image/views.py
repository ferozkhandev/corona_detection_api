from django.core.files import File
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from image import corona_detector
from rest_framework.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_401_UNAUTHORIZED
)

# Create your views here.
class ImageUpload(APIView):
    def post(self, request):
        uploaded_file = None
        if request.method == "POST":
            uploaded_file = request.FILES['image']
            fs = FileSystemStorage()
            fs.save(uploaded_file.name, uploaded_file)
            image_path = settings.MEDIA_ROOT+"\\"+uploaded_file.name
            is_corona = corona_detector.image_loader(image_path)
            result = ""
            if is_corona:
                result = "Corona Detected"
            else:
                result = "Congratulations! You are safe."
        return Response(result, status=HTTP_201_CREATED)