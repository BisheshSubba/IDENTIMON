from django.shortcuts import render

def home(request):
    return render(request,"home_page.html")

def landing(request):
    return render(request,"landing.html")