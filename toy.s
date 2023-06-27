	.text
	.file	"LLVMDialectModule"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
# %bb.0:                                # %.preheader.preheader
	pushq	%r15
	pushq	%r14
	pushq	%rbx
	movl	$64, %edi
	callq	malloc@PLT
	movq	%rax, %rbx
	movl	$64, %edi
	callq	malloc@PLT
	movq	%rax, %r14
	movl	$64, %edi
	callq	malloc@PLT
	movq	%rax, %r15
	movabsq	$4608533498688228557, %rax      # imm = 0x3FF4CCCCCCCCCCCD
	movq	%rax, (%r15)
	movabsq	$4613037098315599053, %rcx      # imm = 0x4004CCCCCCCCCCCD
	movq	%rcx, 8(%r15)
	movabsq	$4616414798036126925, %rcx      # imm = 0x4010CCCCCCCCCCCD
	movq	%rcx, 16(%r15)
	movabsq	$4607632778762754458, %rcx      # imm = 0x3FF199999999999A
	movq	%rcx, 24(%r15)
	movabsq	$4611235658464650854, %rdx      # imm = 0x3FFE666666666666
	movq	%rdx, 32(%r15)
	movabsq	$4612136378390124954, %rdx      # imm = 0x400199999999999A
	movq	%rdx, 40(%r15)
	movabsq	$4612811918334230528, %rdx      # imm = 0x4004000000000000
	movq	%rdx, 48(%r15)
	movabsq	$4620355447710076109, %rdx      # imm = 0x401ECCCCCCCCCCCD
	movq	%rdx, 56(%r15)
	movabsq	$4619792497756654797, %rdx      # imm = 0x401CCCCCCCCCCCCD
	movq	%rdx, 56(%r14)
	movabsq	$4612361558371493478, %rdx      # imm = 0x4002666666666666
	movq	%rdx, 48(%r14)
	movabsq	$4611911198408756429, %rdx      # imm = 0x4000CCCCCCCCCCCD
	movq	%rdx, 40(%r14)
	movabsq	$4607182418800017408, %rdx      # imm = 0x3FF0000000000000
	movq	%rdx, 32(%r14)
	movq	%rax, 24(%r14)
	movabsq	$4616977747989548237, %rax      # imm = 0x4012CCCCCCCCCCCD
	movq	%rax, 16(%r14)
	movabsq	$4612586738352862003, %rax      # imm = 0x4003333333333333
	movq	%rax, 8(%r14)
	movq	%rcx, (%r14)
	movupd	(%r15), %xmm0
	movupd	16(%r15), %xmm1
	movupd	16(%r14), %xmm2
	addpd	%xmm1, %xmm2
	movupd	(%r14), %xmm1
	addpd	%xmm0, %xmm1
	movupd	%xmm2, 16(%rbx)
	movupd	%xmm1, (%rbx)
	movupd	32(%r15), %xmm0
	movupd	48(%r15), %xmm1
	movupd	32(%r14), %xmm2
	addpd	%xmm0, %xmm2
	movupd	48(%r14), %xmm0
	addpd	%xmm1, %xmm0
	movupd	%xmm0, 48(%rbx)
	movupd	%xmm2, 32(%rbx)
	movsd	(%rbx), %xmm0                   # xmm0 = mem[0],zero
	movl	$frmt_spec, %edi
	movb	$1, %al
	callq	printf@PLT
	movsd	8(%rbx), %xmm0                  # xmm0 = mem[0],zero
	movl	$frmt_spec, %edi
	movb	$1, %al
	callq	printf@PLT
	movsd	16(%rbx), %xmm0                 # xmm0 = mem[0],zero
	movl	$frmt_spec, %edi
	movb	$1, %al
	callq	printf@PLT
	movsd	24(%rbx), %xmm0                 # xmm0 = mem[0],zero
	movl	$frmt_spec, %edi
	movb	$1, %al
	callq	printf@PLT
	movsd	32(%rbx), %xmm0                 # xmm0 = mem[0],zero
	movl	$frmt_spec, %edi
	movb	$1, %al
	callq	printf@PLT
	movsd	40(%rbx), %xmm0                 # xmm0 = mem[0],zero
	movl	$frmt_spec, %edi
	movb	$1, %al
	callq	printf@PLT
	movsd	48(%rbx), %xmm0                 # xmm0 = mem[0],zero
	movl	$frmt_spec, %edi
	movb	$1, %al
	callq	printf@PLT
	movsd	56(%rbx), %xmm0                 # xmm0 = mem[0],zero
	movl	$frmt_spec, %edi
	movb	$1, %al
	callq	printf@PLT
	movq	%r15, %rdi
	callq	free@PLT
	movq	%r14, %rdi
	callq	free@PLT
	movq	%rbx, %rdi
	popq	%rbx
	popq	%r14
	popq	%r15
	jmp	free@PLT                        # TAILCALL
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
                                        # -- End function
	.type	frmt_spec,@object               # @frmt_spec
	.section	.rodata,"a",@progbits
frmt_spec:
	.asciz	"%f "
	.size	frmt_spec, 4

	.type	.Lconstant_1,@object            # @constant_1
	.p2align	4, 0x0
.Lconstant_1:
	.quad	0x3ff199999999999a              # double 1.1000000000000001
	.quad	0x4003333333333333              # double 2.3999999999999999
	.quad	0x4012cccccccccccd              # double 4.7000000000000002
	.quad	0x3ff4cccccccccccd              # double 1.3
	.quad	0x3ff0000000000000              # double 1
	.quad	0x4000cccccccccccd              # double 2.1000000000000001
	.quad	0x4002666666666666              # double 2.2999999999999998
	.quad	0x401ccccccccccccd              # double 7.2000000000000002
	.size	.Lconstant_1, 64

	.type	.Lconstant_0,@object            # @constant_0
	.p2align	4, 0x0
.Lconstant_0:
	.quad	0x3ff4cccccccccccd              # double 1.3
	.quad	0x4004cccccccccccd              # double 2.6000000000000001
	.quad	0x4010cccccccccccd              # double 4.2000000000000002
	.quad	0x3ff199999999999a              # double 1.1000000000000001
	.quad	0x3ffe666666666666              # double 1.8999999999999999
	.quad	0x400199999999999a              # double 2.2000000000000002
	.quad	0x4004000000000000              # double 2.5
	.quad	0x401ecccccccccccd              # double 7.7000000000000001
	.size	.Lconstant_0, 64

	.section	".note.GNU-stack","",@progbits
