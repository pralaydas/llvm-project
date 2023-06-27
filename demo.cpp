typedef float v8f __attribute__ ((vector_size(32)));

v8f foo(){
    v8f a = {1.5,2,3,4,5,6,7,8};
    v8f b = {2.5,1,3,4,5,6,7,8};
    v8f c = a+b;
    return c;
}