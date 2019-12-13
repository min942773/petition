from konlpy.tag import Kkma 

kkma = Kkma()
def morph(input_data) : #형태소 분석 
    preprocessed = kkma.pos(input_data) 
    print(preprocessed)

morph("강남, 홍대, 이태원 등등 서울과 지방들의 번화가에는 많은 인터넷방송인들이 셀카봉에 카메라나 휴대폰을 끼고 여기저기 돌아다니며 라이브 방송을 합니다")