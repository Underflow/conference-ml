all:
	pandoc -t beamer -s slides.md --toc -V theme:Rochester -o slides.pdf
tex:
	pandoc -t beamer -s slides.md --toc -V theme:Rochester -o slides.tex

clean:
	rm slides.pdf
