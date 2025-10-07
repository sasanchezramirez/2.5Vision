provider "aws" {
    region = var.aws_region
}

resource "aws_vpc" "main" {
    cidr_block = "10.0.0.0/16"
    tags = {
        Name = "2.5Vision-VPC"
    }
}

resource "aws_subnet" "public" {
    vpc_id = aws_vpc.main.vpc_id
    cidr_block = "10.0.1.0/24"
    map_public_ip_on_launch = true
    tags = {
        Name = "2.5Vision-Public-Subnet"
    }
}

resource "aws_security_group" "server_sg" {
  name        = "2.5vision-server-sg"
  description = "Allow SSH and HTTP inbound traffic"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # CUIDADO: Abierto a todo internet. Idealmente, aquí iría tu IP.
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_s3_bucket" "vision_data" {
  bucket = "2-5-vision-data-bucket-${random_id.bucket_suffix.hex}" # Nombre único para el bucket

  lifecycle_rule {
    id      = "log"
    enabled = true

    expiration {
      days = 30 
    }
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 8
}

resource "aws_instance" "ml_server" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.micro"
  subnet_id     = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.server_sg.id]

  tags = {
    Name    = "2.5Vision-ML-Server"
    Project = "2.5Vision"
  }
}