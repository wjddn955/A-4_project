from django.shortcuts import render

# Create your views here.

def index(request):
    return render(request, "main/index.html")

def test(request):
    return render(request, "main/desktop-app.html")

def memo(request):
    return render(request, "main/memo.html")