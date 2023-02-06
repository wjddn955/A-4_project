from django.shortcuts import render, redirect
from .models import Diary_content 
from django.http import HttpResponse

# Create your views here.

def index(request):
    return render(request, "main/index.html")

def test(request):
    return render(request, "main/desktop-app.html")

def memo(request):
    return render(request, "main/memo.html")

def result(request):
    if request.method == 'POST' and request.POST['diary_input'] != '': # 빈 값 x
        content = Diary_content()
        content.para = request.POST['diary_input']
        content.save()
    else:
        return HttpResponse("<script>alert('빈 값입니다, 다시 입력 해주세요!');location.href='/test';</script>")
        #return redirect('test')

    return redirect('test')